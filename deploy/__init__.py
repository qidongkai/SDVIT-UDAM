"""
Edge deployment utilities.

This package exposes the ONNX export, operator compatibility check,
SNPE / TensorRT engine builders, latency and power measurement helpers,
and the statistical validator used to report the deployment metrics
required by Section 3.5.5 of the thesis.
"""
from .export_onnx import export
from .check_ops import check
from .build_snpe import build as build_snpe
from .build_tensorrt import build as build_tensorrt
from .measure_latency import measure as measure_latency
from .measure_power import measure as measure_power
from .stats_validate import validate

__all__ = [
    "export",
    "check",
    "build_snpe",
    "build_tensorrt",
    "measure_latency",
    "measure_power",
    "validate",
]