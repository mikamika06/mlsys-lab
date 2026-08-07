import ref

def check(workdir):
    from roofline.model import compute_decode_roofline
    out = {"predictions_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_decode_roofline(cfg, 4, 900.0 * 10**9, 300.0 * 10**12)
        try:
            got = compute_decode_roofline(cfg, 4, 900.0 * 10**9, 300.0 * 10**12)
            if got is not None and abs(got - want) / (want + 1e-9) < 1e-4:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"config {i}: got {got}, reference {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"config {i} raised {type(e).__name__}"
    out["predictions_matched"] = float(ok)
    return out
