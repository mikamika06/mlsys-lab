import ref

def check(workdir):
    from roofline.spec import compute_ridge_point

    out = {"ridge_points_matched": 0.0, "configs": float(len(ref.SPECS))}
    ok = 0
    for i, spec in enumerate(ref.SPECS):
        want = ref.compute_ridge_point(spec["peak_flops"], spec["peak_bandwidth"])
        try:
            got = compute_ridge_point(spec["peak_flops"], spec["peak_bandwidth"])
        except Exception:
            got = -1.0
        if abs(got - want) < 1e-5:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"spec {i}: got {got}, reference {want}"
    out["ridge_points_matched"] = float(ok)
    return out
