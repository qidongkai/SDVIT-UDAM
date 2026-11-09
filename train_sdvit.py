"""
SDViT training script.

Implements the supervised training of the Static-Dynamic collaborative
Vision Transformer described in Section 3.2.1 of the thesis.

Training protocol (Section 3.5.1 and 4.1.1):
    - AdamW, lr = 1e-4, weight_decay = 1e-5
    - cosine annealing to lr_min = 1e-6 over 120 epochs
    - per-device batch size 8, gradient accumulation 2, effective 16
    - AMP with GradScaler
    - gradient clipping at 1.0
    - scene-aware sampler on AR-Private-20K
    - ten pre-declared seeds

Loss (Eq 3.5):
    L = L_seg + lambda_sd * L_sd

The self-distillation loss is computed layer by layer, using the
deepest unpruned feature as the target for each pruned intermediate
feature (Eq 3.4 / Eq 3.18).
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from configs.loader import load_config
from data.dataset import ARSegDataset
from data.sampler import SceneAwareSampler
from data.transforms import build_train_transform
from eval.metrics import SegMetrics
from losses.sd_loss import (
    compute_reliability_mask,
    self_distill_loss,
    total_sdvit_loss,
)
from models.sdvit import SDViT
from utils.logger import get_logger
from utils.seed import set_seed


def train_one_epoch(model, loader, opt, scaler, cfg, device, logger, epoch):
    """
    Run one full epoch of supervised SDViT training.

    Returns:
        The mean intersection-over-union on the training split.
    """
    model.train()
    metric = SegMetrics(cfg["model"]["num_classes"])
    accum = cfg["train"]["grad_accum_steps"]
    img_size = cfg["model"]["img_size"]

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(enabled=cfg["train"]["amp"]):
            feat, mid_feats, f_deep = model(x, return_aux=True)

            B = feat.shape[0]
            grid = int(feat.shape[1] ** 0.5)
            f2d = feat.transpose(1, 2).reshape(B, -1, grid, grid)
            logits = F.interpolate(
                model.head(f2d),
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )

            # Layer-wise self-distillation (Eq 3.4 / Eq 3.18).
            l_sd = torch.tensor(0.0, device=device)
            for layer_idx, mid_feat, mid_mask, _ in mid_feats:
                proj = model.projectors[str(layer_idx)](mid_feat, grid)
                rel_mask = compute_reliability_mask(proj, f_deep)
                l_sd = l_sd + self_distill_loss(f_deep, proj, rel_mask)

            loss, l_seg, l_sd_val = total_sdvit_loss(
                logits, y, l_sd, cfg["model"]["lambda_sd"]
            )
            loss = loss / accum

        scaler.scale(loss).backward()

        if (step + 1) % accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["train"]["clip_grad"]
            )
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        metric.update(
            logits.argmax(dim=1).detach().cpu().numpy(),
            y.detach().cpu().numpy(),
        )

        if (step + 1) % 50 == 0:
            logger.info(
                f"[epoch {epoch}] step {step + 1}/{len(loader)} "
                f"seg={l_seg.item():.4f} sd={l_sd_val.item():.4f}"
            )

    return metric.miou()


def main(cfg_path, seed):
    set_seed(seed)
    cfg = load_config(cfg_path)

    os.makedirs("runs", exist_ok=True)
    ckpt_path = os.path.join("runs", f"sdvit_seed{seed}.pth")
    logger = get_logger(
        name=f"sdvit.seed{seed}",
        log_path=os.path.join("runs", f"sdvit_seed{seed}.log"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SDViT(cfg["model"]).to(device)

    dataset = ARSegDataset(
        root=cfg["data"]["root_cityscapes"],
        split="train",
        transform=build_train_transform(cfg["model"]["img_size"]),
        unlabeled=False,
        domain="cityscapes",
    )
    sampler = SceneAwareSampler(
        scene_ids=dataset.scene_ids,
        batch_size=cfg["train"]["batch_size_per_gpu"],
        shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size_per_gpu"],
        sampler=sampler,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    opt = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    sched = CosineAnnealingLR(
        opt,
        T_max=cfg["train"]["epochs"],
        eta_min=cfg["train"]["min_lr"],
    )
    scaler = GradScaler()

    best_miou = 0.0
    for epoch in range(cfg["train"]["epochs"]):
        miou = train_one_epoch(
            model, loader, opt, scaler, cfg, device, logger, epoch
        )
        sched.step()
        logger.info(f"[epoch {epoch}] train mIoU = {miou:.4f}")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"saved new best checkpoint to {ckpt_path}")

    logger.info(f"training finished, best train mIoU = {best_miou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.cfg, args.seed)