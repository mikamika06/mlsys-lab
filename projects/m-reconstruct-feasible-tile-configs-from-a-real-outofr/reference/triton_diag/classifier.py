def classify_errors(error_strings: list[str]) -> list[str]:
    results = []
    for s in error_strings:
        if "OutOfResources" in s or "out of" in s.lower():
            results.append("OutOfResources")
        elif "CompilationError" in s or "pointer type" in s:
            results.append("CompilationError")
        elif "Out-of-memory" in s or "workspace" in s:
            results.append("OutOfMemory")
        else:
            results.append("InternalError")
    return results
