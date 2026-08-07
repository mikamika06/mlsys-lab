def classify_triton_error(error_msg: str) -> str:
    m = error_msg.lower()
    if "unexpected shape" in m or "shape mismatch" in m or "incompatible shape" in m:
        return "SHAPE_MISMATCH"
    if "exceeds max_batch_size" in m or "batch size exceeds" in m or "exceeds max batch" in m:
        return "MAX_BATCH_EXCEEDED"
    if "profile" in m or "out of bounds" in m or "min/max shape" in m:
        return "PROFILE_OUT_OF_BOUNDS"
    if "failed to allocate" in m or "out of memory" in m or "cuda error out of memory" in m:
        return "OOM_DYNAMIC_SHAPE"
    if "unbounded" in m or "must specify" in m or "explicit dimension" in m:
        return "UNBOUNDED_DYNAMIC_SHAPE"
    return "UNKNOWN_ERROR"
