def classify_error(err_str):
    if "OpCode not supported" in err_str:
        return "unsupported_op"
    if "dynamic dimensions" in err_str:
        return "dynamic_shape"
    if "OutOfMemoryError" in err_str:
        return "out_of_memory"
    if "Missing tensor" in err_str:
        return "missing_tensor"
    return "unknown"
