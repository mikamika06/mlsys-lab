def classify_traceback(parsed_dump: dict) -> str:
    err = parsed_dump.get("error", "")
    if not err:
        return "SUCCESS"
    if "cudart" in err.lower() or "libcudart" in err.lower():
        return "MISSING_CUDART"
    if "version" in err.lower() or "mismatch" in err.lower():
        return "CUDA_VERSION_MISMATCH"
    if "not found" in err.lower() or "shared object" in err.lower():
        return "MISSING_SHARED_OBJECT"
    return "UNKNOWN_ERROR"
