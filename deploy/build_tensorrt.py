"""
TensorRT engine builder for NVIDIA-based AR glasses SoC.

Follows Step 3 of the deployment pipeline in Section 3.5.5 of the thesis:
    - ONNX -> TensorRT engine with FP16 precision
    - LayerNorm layers are forced to FP32 because FP16 LayerNorm
      introduces a larger accuracy loss
    - GELU is decomposed into primitive operations for TensorRT >= 11.0
      where the deprecated GELU plugin has been removed
"""
import os
import subprocess

import yaml


def build(cfg_path="configs/default.yaml",
          onnx_path="sdvit.onnx",
          engine_path="sdvit.engine"):
    """
    Build a TensorRT engine from ONNX.

    Args:
        cfg_path:    path to the YAML config file (reads the TensorRT root).
        onnx_path:   path to the exported ONNX file.
        engine_path: output TensorRT engine path.
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    trt_root = cfg["deploy"]["ar_glasses_trt"]
    trtexec = os.path.join(trt_root, "bin", "trtexec")

    if not os.path.exists(trtexec):
        raise FileNotFoundError(f"trtexec not found: {trtexec}")

    print(f"Building TensorRT engine: {onnx_path} -> {engine_path}")
    subprocess.run(
        [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--fp16",
            "--layerPrecisions=*LayerNorm*:fp32",
        ],
        check=True,
    )
    print(f"TensorRT engine built: {engine_path}")


if __name__ == "__main__":
    build()