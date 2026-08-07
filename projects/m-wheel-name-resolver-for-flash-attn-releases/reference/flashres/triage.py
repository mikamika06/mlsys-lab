def triage_traceback(traceback_text: str) -> str:
    lower = traceback_text.lower()
    if "not a supported wheel" in lower or "is not a supported wheel on this platform" in lower:
        return "wheel_tag_mismatch"
    if "nvcc" in lower or "cuda error" in lower or "no kernel image is available" in lower:
        return "cuda_version_mismatch"
    if "undefined symbol" in lower or "importError" in traceback_text:
        return "abi_incompatibility"
    if "out of memory" in lower or "cuda out of memory" in lower:
        return "out_of_memory"
    if "ninja" in lower or "failed to build" in lower:
        return "build_compilation_failure"
    return "unknown_error"
