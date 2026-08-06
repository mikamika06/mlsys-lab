import ref

def check(workdir):
    from skew.checker import detect_skew
    from skew.runtime import get_supported_versions
    out = {"skew_matched": 0.0}
    caps = get_supported_versions()
    ok = 0
    for i, h in enumerate(ref.HEADERS):
        meta = ref.parse_metadata(h)
        want = ref.detect_skew(meta, caps)
        got = detect_skew(meta, caps)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"header {i}: got {got}, reference {want}"
    out["skew_matched"] = 1.0 if ok == len(ref.HEADERS) else 0.0
    return out
