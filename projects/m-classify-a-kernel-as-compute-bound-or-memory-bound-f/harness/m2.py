import ref


def check(workdir):
    from roofline.classify import classify_kernel

    out = {"classification_match": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.classify_kernel(
            cfg["flops"], cfg["bytes_transferred"], cfg["peak_flops"], cfg["peak_bandwidth"]
        )
        got = classify_kernel(
            cfg["flops"], cfg["bytes_transferred"], cfg["peak_flops"], cfg["peak_bandwidth"]
        )
        if got[0] == want[0] and abs(got[1] - want[1]) < 1e-6 and abs(got[2] - want[2]) < 1e-6:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["classification_match"] = float(ok)
    return out
