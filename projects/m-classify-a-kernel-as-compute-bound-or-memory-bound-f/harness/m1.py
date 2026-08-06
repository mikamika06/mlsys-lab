import ref


def check(workdir):
    from roofline.intensity import compute_intensity

    out = {"intensity_match": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_intensity(cfg["flops"], cfg["bytes_transferred"])
        got = compute_intensity(cfg["flops"], cfg["bytes_transferred"])
        if abs(got - want) < 1e-6:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["intensity_match"] = float(ok)
    return out
