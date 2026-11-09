"""
Ablation study for SDViT.

Implements the six groups defined in Table 4-14 of the thesis:
    (i)   full SDViT (adaptive pruning + self-distillation)
    (ii)  static baseline (no pruning, no self-distillation)
    (iii) pruning only (no self-distillation)
    (iv)  self-distillation only (no pruning)
    (v)   full SDViT with lambda_sd = 0
    (vi)  full SDViT with a fixed pruning threshold

Groups (v) and (vi) are supplementary controls that isolate the
contribution of specific design choices within the self-distillation and
pruning mechanisms.

Each group is evaluated on the AR-Private-20K validation split, using
the ten-seed protocol of Section 4.1.4.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from configs.loader import load_config
from data.dataset import ARSegDataset
from data.transforms import build_train_transform
from eval.metrics import SegMetrics
from models.sdvit import SDViT
from utils.logger import get_logger
from utils.seed import set_seed


# Six ablation groups from Table 4-14.
VARIANTS = {
    "full":        {"prune": True,  "sd": True,  "thr": "adaptive", "lambda_sd": 0.3},
    "no_prune":    {"prune": False, "sd": False, "thr": "adaptive", "lambda_sd": 0.3},
    "prune_only":  {"prune": True,  "sd": False, "thr": "adaptive", "lambda_sd": 0.0},
    "sd_only":     {"prune": False, "sd": True,  "thr": "adaptive", "lambda_sd": 0.3},
    "lambda_zero": {"prune": True,  "sd": True,  "thr": "adaptive", "lambda_sd": 0.0},
    "fixed_thr":   {"prune": True,  "sd": True,  "thr": "fixed",    "lambda_sd": 0.3},
}


def build_variant_config(base_cfg, variant_name):
    """
    Clone the base config and override the fields that define the
    variant. The returned config is a deep copy so the base config is
    not modified.
    """
    import copy

    cfg = copy.deepcopy(base_cfg)
    v = VARIANTS[variant_name]

    if not v["prune"]:
        cfg["model"]["prune_layers"] = []
    cfg["model"]["lambda_sd"] = v["lambda_sd"]
    cfg["model"]["use_fixed_thr"] = (v["thr"] == "fixed")
    if cfg["model"]["use_fixed_thr"]:
        cfg["model"]["fixed_ratio"] = 0.58
    return cfg


def evaluate_variant(cfg_path, variant_name, seed):
    """
    Load a trained checkpoint and evaluate it under the given ablation
    configuration. The checkpoint is assumed to exist at
    runs/sdvit_seed{seed}.pth.
    """
    set_seed(seed)
    base_cfg = load_config(cfg_path)
    cfg = build_variant_config(base_cfg, variant_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SDViT(cfg["model"]).to(device)

    ckpt = os.path.join("runs", f"sdvit_seed{seed}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    dataset = ARSegDataset(
        root=cfg["data"]["root_ar"],
        split="val",
        transform=build_train_transform(cfg["model"]["img_size"]),
        unlabeled=False,
        domain="ar",
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size_per_gpu"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    metric = SegMetrics(cfg["model"]["num_classes"])
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            metric.update(
                logits.argmax(dim=1).cpu().numpy(),
                y.cpu().numpy(),
            )

    miou = metric.miou()
    print(f"[{variant_name}] seed={seed} mIoU={miou:.4f}")
    return miou


def main(cfg_path, variant, seed):
    logger = get_logger(
        name=f"ablation.{variant}",
        log_path=os.path.join("runs", f"ablation_{variant}.log"),
    )
    miou = evaluate_variant(cfg_path, variant, seed)
    logger.info(f"[{variant}] seed={seed} mIoU={miou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/default.yaml")
    parser.add_argument("--variant", required=True, choices=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.cfg, args.variant, args.seed)