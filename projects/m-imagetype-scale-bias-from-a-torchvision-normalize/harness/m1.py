import ref


def check(workdir):
    from core_image.normalize import compute_scale_bias

    out = {"mappings_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want_scale, want_bias = ref.compute_scale_bias(cfg["mean"], cfg["std"])
        try:
            got_scale, got_bias = compute_scale_bias(cfg["mean"], cfg["std"])
        except Exception:
            got_scale, got_bias = None, None

        if got_scale == want_scale and got_bias == want_bias:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got scale {got_scale}, reference {want_scale}"
    out["mappings_matched"] = float(ok)
    return out
