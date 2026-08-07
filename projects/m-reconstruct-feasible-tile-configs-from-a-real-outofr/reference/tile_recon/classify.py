def classify_errors(error_strings):
    results = []
    for err in error_strings:
        lower = err.lower()
        if "shared memory" in lower or "smem" in lower:
            results.append("SHARED_MEMORY_EXCEEDED")
        elif "register" in lower or "regs" in lower:
            results.append("REGISTER_PRESSURE")
        elif "grid" in lower or "block size" in lower:
            results.append("GRID_SIZE_LIMIT")
        else:
            results.append("UNKNOWN_OR_SYNTAX")
    return results
