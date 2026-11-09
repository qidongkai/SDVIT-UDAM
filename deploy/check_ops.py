"""
Operator compatibility check for the exported ONNX graph.

Follows Step 2 of the deployment pipeline in Section 3.5.5 of the thesis:
the exported graph is loaded with onnxruntime and its operator set is
inspected to confirm that LayerNormalization, MatMul, Softmax, Conv and
GELU are present. GELU may be decomposed into primitive operations for
TensorRT versions where the deprecated GELU plugin has been removed.
"""
import onnxruntime as ort


REQUIRED_OPS = {"MatMul", "Softmax", "LayerNormalization", "Conv", "Gelu"}
OPTIONAL_OPS = {"Erf"}  # GELU decomposition path


def check(model_path="sdvit.onnx"):
    """
    Inspect the ONNX graph and report operator compatibility.

    Args:
        model_path: path to the exported ONNX file.

    Returns:
        True if all required operators are present, False otherwise.
    """
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    ops = {node.op_type for node in sess.get_modelmeta().graph.node}

    print("Operators present:")
    for op in sorted(ops):
        print(f"  - {op}")

    missing = REQUIRED_OPS - ops
    if missing:
        print(f"Missing required operators: {sorted(missing)}")
        print("Compatibility check: WARN")
        return False

    print("Compatibility check: PASS")
    if "Gelu" not in ops and "Erf" in ops:
        print("GELU is decomposed into primitive operations (Erf-based path).")
    return True


if __name__ == "__main__":
    check()