def classify_error(error_string):
    s = error_string.lower()
    if "dynamic shape" in s or "shape mismatch" in s:
        return "DYNAMIC_SHAPE_ERROR"
    if "out of memory" in s or "cuda oom" in s:
        return "OOM_ERROR"
    if "concurrency" in s or "queue overflow" in s:
        return "CONCURRENCY_CEILING_EXCEEDED"
    return "UNKNOWN_ERROR"
