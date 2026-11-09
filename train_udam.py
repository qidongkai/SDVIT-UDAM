"""
UDAM cross-domain adaptation training script.

Implements the source-free unsupervised domain adaptation of Section
3.2.2 of the thesis.

Key protocol points:
    - the teacher is frozen; only the student is updated
    - the source-domain feature bank (800 vectors, ~2.5 MB) is built once
      from the frozen teacher and never stores raw source images
    - the target-domain data are unlabeled; the K-view multi-view
      transform supplies the self-supervised consistency signal
    - pseudo-label reliability is controlled by the four-stage filter in
      losses/reliability.py
    - the adaptive domain-gap coefficient gamma is computed from MMD and
      clamped to a floor of 0.2 (Section 4.5.3)

Loss (Eq 3.9):
    L_UDAM = rho * (gamma * L_kd + (1 - gamma) * L_ss)
"""
import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from configs.loader import load_config
from data.dataset import ARSegDataset
from data.transforms import MultiViewTransform, build_train_transform
from losses.reliability import (
    augmentation_consistency,
    sample_filter,
)
from losses.udam_loss import udam_total_loss
from models.sdvit import SDViT
from models.teacher import SegFormerTeacher
from utils.logger import get_logger
from utils.seed import set_seed


def build_feature_bank(teacher, source_loader, num_classes=16,
                       per_class=50, max_vectors=800, device="cuda"):
    """
    Build the source-domain feature bank.

    For each of the 16 unified categories, up to `per_class` teacher
    feature vectors are collected from the source training set. The
    resulting bank is stored as a fixed tensor and is never invertible to
    an image, satisfying the source-free requirement of Section 4.3.0.

    Args:
        teacher:       frozen SegFormer-B5 teacher.
        source_loader: labeled source-domain dataloader.
        num_classes:   number of unified classes.
        per_class:     number of vectors per class.
        max_vectors:   hard cap on the bank size.
        device:        device for the teacher.

    Returns:
        [N, D] tensor with N <= max_vectors.
    """
    bank = []
    counts = [0] * num_classes

    teacher.eval()
    with torch.no_grad():
        for x, y in source_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            feats = teacher.stage_features(x)[-1]   # [B, C, h, w]
            pooled = feats.mean(dim=[2, 3])         # [B, C]

            for c in range(num_classes):
                if counts[c] >= per_class:
                    continue
                # Pixels of class c in this batch, if any.
                mask = (y == c).any(dim=[1, 2])
                if mask.any():
                    take = pooled[mask]
                    bank.append(take)
                    counts[c] += take.shape[0]

    if len(bank) == 0:
        raise RuntimeError("feature bank is empty; check the source loader")

    bank = torch.cat(bank, dim=0)
    bank = bank[:max_vectors]
    return bank


def main(cfg_path, seed):
    set_seed(seed)
    cfg = load_config(cfg_path)

    os.makedirs("runs", exist_ok=True)
    student_ckpt = os.path.join("runs", f"sdvit_seed{seed}.pth")
    out_ckpt = os.path.join("runs", f"sdvit_udam_seed{seed}.pth")
    logger = get_logger(
        name=f"udam.seed{seed}",
        log_path=os.path.join("runs", f"udam_seed{seed}.log"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = SDViT(cfg["model"]).to(device)
    if os.path.exists(student_ckpt):
        student.load_state_dict(torch.load(student_ckpt, map_location=device))
        logger.info(f"loaded student checkpoint from {student_ckpt}")
    else:
        logger.warning(f"{student_ckpt} not found, starting from scratch")

    teacher = SegFormerTeacher().to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Source loader is only used to build the feature bank.
    source_dataset = ARSegDataset(
        root=cfg["data"]["root_cityscapes"],
        split="train",
        transform=build_train_transform(cfg["model"]["img_size"]),
        unlabeled=False,
        domain="cityscapes",
    )
    source_loader = DataLoader(
        source_dataset,
        batch_size=cfg["train"]["batch_size_per_gpu"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    bank = build_feature_bank(
        teacher,
        source_loader,
        num_classes=cfg["model"]["num_classes"],
        per_class=cfg["udam"]["per_class_bank"],
        max_vectors=cfg["udam"]["feature_bank_size"],
        device=device,
    ).to(device)
    logger.info(f"feature bank built with shape {tuple(bank.shape)}")

    # Target loader with K-view multi-view transform.
    mv = MultiViewTransform(
        size=cfg["model"]["img_size"],
        K=cfg["udam"]["K_views"],
    )
    target_dataset = ARSegDataset(
        root=cfg["data"]["root_ar"],
        split="train",
        transform=mv,
        unlabeled=True,
        domain="ar",
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=cfg["train"]["batch_size_per_gpu"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    opt = AdamW(
        student.parameters(),
        lr=cfg["train"]["lr"] * 0.5,
        weight_decay=cfg["train"]["weight_decay"],
    )

    for epoch in range(cfg["train"]["epochs"]):
        student.train()
        stats = []

        for views, _ in target_loader:
            views = views.to(device, non_blocking=True)
            B, K, C, H, W = views.shape

            with torch.no_grad():
                teacher_soft = teacher.soft_labels(views[:, 0])
                consistency = augmentation_consistency(
                    teacher, views[:, 0], K=K
                )

            feats = []
            probs = []
            for k in range(K):
                f_k = student.forward_features(views[:, k])
                feats.append(f_k.mean(dim=1))
                logits_k = student(views[:, k])
                probs.append(logits_k.softmax(dim=1))

            f_t = feats[0]
            p_t = probs[0]
            f_aug_list = feats[1:]

            s_mask = sample_filter(p_t, cfg["udam"]["tau_H"])

            loss, info = udam_total_loss(
                teacher_soft, p_t, f_t, f_aug_list, bank, cfg
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                student.parameters(), cfg["train"]["clip_grad"]
            )
            opt.step()

            info["consistency"] = consistency.mean().item()
            info["s_mask"] = s_mask.mean().item()
            stats.append(info)

        if stats:
            avg = {k: sum(d[k] for d in stats) / len(stats) for k in stats[0]}
            logger.info(f"[udam epoch {epoch}] {avg}")

    torch.save(student.state_dict(), out_ckpt)
    logger.info(f"saved adapted student to {out_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.cfg, args.seed)