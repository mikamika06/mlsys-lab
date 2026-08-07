CAST_OPS = {"matmul", "linear", "conv1d", "conv2d", "bmm", "addmm"}
KEEP_FP32_OPS = {"softmax", "layer_norm", "group_norm", "cross_entropy", "exp", "log", "pow", "sum"}
PROMOTE_OPS = {"add", "sub", "mul", "div", "cat"}


def predict_autocast_action(op_name: str, target_dtype: str = "fp16") -> str:
    op_clean = op_name.lower().strip()
    if op_clean in CAST_OPS:
        return "cast"
    if op_clean in KEEP_FP32_OPS:
        return "keep_fp32"
    if op_clean in PROMOTE_OPS:
        return "promote"
    return "keep_fp32"
