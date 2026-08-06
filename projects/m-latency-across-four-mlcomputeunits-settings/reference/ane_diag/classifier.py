ALLOWED_OPS = {"conv2d", "matmul", "depthwise_conv2d", "relu", "add"}
MAX_SPATIAL_DIM = 4096


def is_ane_eligible(op_spec):
    """Predict ANE execution eligibility for a given operator specification."""
    op_type = op_spec.get("op_type")
    if op_type not in ALLOWED_OPS:
        return False

    dtype = op_spec.get("dtype")
    if dtype not in ("float16", "int8"):
        return False

    shape = op_spec.get("shape", [])
    if len(shape) != 4:
        return False

    batch, channels, height, width = shape
    if batch != 1:
        return False

    if height > MAX_SPATIAL_DIM or width > MAX_SPATIAL_DIM:
        return False

    if op_type in ("conv2d", "depthwise_conv2d"):
        k_height = op_spec.get("kernel_height", 1)
        k_width = op_spec.get("kernel_width", 1)
        if k_height > 15 or k_width > 15:
            return False

    if op_type == "matmul":
        if channels % 16 != 0:
            return False

    return True
