"""
SNPE engine builder for Qualcomm Snapdragon 8 Gen 2.

Follows Step 3 of the deployment pipeline in Section 3.5.5 of the thesis:
    - ONNX -> DLC conversion using snpe-onnx-to-dlc
    - INT8 post-training quantization using snpe-dlc-quantize
      with calibration data drawn from the AR-Private-20K validation set
    - the resulting DLC is executed through the SNPE runtime on the
      Hexagon DSP
"""
import os
import subprocess

import yaml


def build(cfg_path="configs/default.yaml",
          onnx_path="sdvit.onnx",
          dlc_path="sdvit.dlc",
          calib_list="calib.txt"):
    """
    Convert ONNX to DLC and quantize to INT8.

    Args:
        cfg_path:   path to the YAML config file (reads the SNPE root).
        onnx_path:  path to the exported ONNX file.
        dlc_path:   output DLC file path.
        calib_list: text file listing the calibration inputs.
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    snpe_root = cfg["deploy"]["snapdragon_snpe"]
    converter = os.path.join(snpe_root, "bin", "snpe-onnx-to-dlc")
    quantizer = os.path.join(snpe_root, "bin", "snpe-dlc-quantize")

    if not os.path.exists(converter):
        raise FileNotFoundError(f"SNPE converter not found: {converter}")

    print(f"[Step 1/2] ONNX -> DLC: {onnx_path} -> {dlc_path}")
    subprocess.run([converter, "-i", onnx_path, "-o", dlc_path], check=True)

    int8_dlc = dlc_path.replace(".dlc", "_int8.dlc")
    print(f"[Step 2/2] INT8 quantization: {dlc_path} -> {int8_dlc}")
    subprocess.run(
        [
            quantizer,
            "--input_dlc", dlc_path,
            "--input_list", calib_list,
            "--output_dlc", int8_dlc,
        ],
        check=True,
    )
    print(f"SNPE INT8 DLC built: {int8_dlc}")


if __name__ == "__main__":
    build()