import ref

def check(workdir):
    from moefit.compare import compare_formats
    out = {"memory_match": 0.0, "overhead_match": 0.0}
    spec = ref.MODELS[0]
    want = ref.compare_formats(spec)
    got = compare_formats(spec)
    m_ok = got.get("mlx", {}).get("bytes") == want["mlx"]["bytes"]
    o_ok = got.get("gguf", {}).get("bytes") == want["gguf"]["bytes"]
    out["memory_match"] = 1.0 if m_ok else 0.0
    out["overhead_match"] = 1.0 if o_ok else 0.0
    if not m_ok or not o_ok:
        out["_note"] = f"got {got}, want {want}"
    return out
