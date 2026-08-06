def classify_snippet(snippet: str) -> str:
    s = snippet.lower()
    if any(x in s for x in ("sm_", "ptx", ".target")):
        return "CUDA"
    if any(x in s for x in ("amdgcn", "s_waitcnt")):
        return "ROCm"
    if "gluon" in s:
        return "Gluon"
    return "Unknown"
