import ref

def check(workdir):
    from roofline.classify import classify_decode

    out = {"classifications_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.classify_decode(cfg["intensity"], cfg["ridge"])
        try:
            got = classify_decode(cfg["intensity"], cfg["ridge"])
        except Exception:
            got = "error"
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["classifications_matched"] = float(ok)
    return out
