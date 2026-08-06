import numpy as np

TARGETS = ["iOS15", "iOS16", "iOS17", "iOS18"]

OP_SUPPORT = {
    "iOS15": {"add", "mul", "conv", "relu"},
    "iOS16": {"add", "mul", "conv", "relu", "gelu", "layernorm"},
    "iOS17": {"add", "mul", "conv", "relu", "gelu", "layernorm", "scaled_dot_product_attention"},
    "iOS18": {"add", "mul", "conv", "relu", "gelu", "layernorm", "scaled_dot_product_attention", "flash_attention"}
}

def validate_target_matrix(target, ops):
    supported = OP_SUPPORT.get(target, set())
    unsupported = [op for op in ops if op not in supported]
    return {"valid": len(unsupported) == 0, "unsupported": unsupported}

def repair_image_input(spec):
    fixed = dict(spec)
    if "scale" not in fixed:
        fixed["scale"] = 1.0 / 255.0
    if "bias" not in fixed:
        fixed["bias"] = [0.0, 0.0, 0.0]
    if "color_space" not in fixed:
        fixed["color_space"] = "RGB"
    fixed["is_image"] = True
    return fixed

def enumerate_opset(model_spec):
    ops = model_spec.get("ops", [])
    freq = {}
    for op in ops:
        freq[op] = freq.get(op, 0) + 1
    opset_version = model_spec.get("opset_version", 2)
    return {"opset_version": opset_version, "frequencies": freq}
