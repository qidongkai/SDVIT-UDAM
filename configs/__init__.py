#!/usr/bin/env python3
import os

ROOT = "sdvit_udam"

INIT_FILES = {
    "__init__.py": '"""SDViT-UDAM: AR 语义分割知识蒸馏框架"""\n__version__ = "0.1.0"\n',

    "configs/__init__.py": '''from .loader import load_config

__all__ = ["load_config"]
''',

    "models/__init__.py": '''from .vit_backbone import PatchEmbed, VarLenAttention, Block
from .dynamic_pruner import TokenScorer, DynamicPruneBlock, var_len_prune
from .projector import ProjectionAlign
from .seg_head import UperNetHead
from .sdvit import SDViT
from .teacher import SegFormerTeacher

__all__ = [
    "PatchEmbed", "VarLenAttention", "Block",
    "TokenScorer", "DynamicPruneBlock", "var_len_prune",
    "ProjectionAlign", "UperNetHead", "SDViT", "SegFormerTeacher",
]
''',

    "losses/__init__.py": '''from .sd_loss import self_distill_loss, total_sdvit_loss, compute_reliability_mask
from .reliability import (
    pixel_filter, sample_filter, augmentation_consistency,
    temporal_stability, adaptive_fallback,
)
from .udam_loss import (
    mmd_rbf, adaptive_gamma, kl_shared, multi_view_consistency, udam_total_loss,
)

__all__ = [
    "self_distill_loss", "total_sdvit_loss", "compute_reliability_mask",
    "pixel_filter", "sample_filter", "augmentation_consistency",
    "temporal_stability", "adaptive_fallback",
    "mmd_rbf", "adaptive_gamma", "kl_shared", "multi_view_consistency",
    "udam_total_loss",
]
''',

    "data/__init__.py": '''from .dataset import ARSegDataset, CITYSCAPES_TO_UNIFIED
from .transforms import build_train_transform, MultiViewTransform
from .sampler import SceneAwareSampler
from .scene_check import check_scene_split

__all__ = [
    "ARSegDataset", "CITYSCAPES_TO_UNIFIED",
    "build_train_transform", "MultiViewTransform",
    "SceneAwareSampler", "check_scene_split",
]
''',

    "eval/__init__.py": '''from .metrics import SegMetrics, pcr, ccr, prr, der
from .cka import (
    cka, stage_mapping, align_spatial, compute_cka_all,
    compute_projection_lstsq, cosine_sim,
)
from .ood import expected_calibration_error, group_by_condition
from .der_extreme import CASES as DER_CASES, der as der_extreme, run_extreme_cases
from .stats import paired_test, holm_within_table, run_table_tests

__all__ = [
    "SegMetrics", "pcr", "ccr", "prr", "der",
    "cka", "stage_mapping", "align_spatial", "compute_cka_all",
    "compute_projection_lstsq", "cosine_sim",
    "expected_calibration_error", "group_by_condition",
    "DER_CASES", "der_extreme", "run_extreme_cases",
    "paired_test", "holm_within_table", "run_table_tests",
]
''',

    "deploy/__init__.py": '''from .export_onnx import export
from .check_ops import check

__all__ = ["export", "check"]
''',

    "utils/__init__.py": '''from .seed import set_seed
from .logger import get_logger
from .vis import feature_heatmap

__all__ = ["set_seed", "get_logger", "feature_heatmap"]
''',
}

for rel_path, content in INIT_FILES.items():
    abs_path = os.path.join(ROOT, rel_path)
    os.makedirs(os.path.dirname(abs_path) if os.path.dirname(abs_path) else ".", exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"created: {abs_path}")

print("done")