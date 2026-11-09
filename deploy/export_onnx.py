"""
ONNX export for SDViT.

Follows Step 1 of the deployment pipeline in Section 3.5.5 of the thesis:
    - static input shape (1, 3, 512, 512)
    - opset 17 (native LayerNormalization support)
    - constant folding enabled
    - dynamic batch axis
"""
import yaml
import torch

from models.sdvit import SDViT


def export(cfg_path, ckpt_path, out_path):
    """
    Export the SDViT student to ONNX.

    Args:
        cfg_path:  path to the YAML config file.
        ckpt_path: path to the PyTorch checkpoint.
        out_path:  output ONNX file path.
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model = SDViT(cfg["model"]).cuda().eval()
    state = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(state, strict=False)

    img_size = cfg["model"]["img_size"]
    dummy = torch.randn(1, 3, img_size, img_size).cuda()

    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=cfg["deploy"]["onnx_opset"],
        input_names=["input"],
        output_names=["logits"],
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"ONNX exported to {out_path}")


if __name__ == "__main__":
    export(
        cfg_path="configs/default.yaml",
        ckpt_path="sdvit_udam_seed0.pth",
        out_path="sdvit.onnx",
    )