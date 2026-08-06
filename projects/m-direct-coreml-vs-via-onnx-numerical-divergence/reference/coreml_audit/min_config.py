import numpy as np
from .divergence import max_abs_error

def minimal_op_config(models):
    allowed_ops = sorted(list({node["op"] for m in models for node in m["nodes"] if node["provider"] == "ANE"}))
    max_err = max(max_abs_error(m["direct"], m["onnx"]) for m in models)
    return {"allowed_ops": allowed_ops, "max_tolerance": max_err * 1.5}
