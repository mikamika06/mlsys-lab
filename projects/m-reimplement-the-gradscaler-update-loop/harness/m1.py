import ref


def check(workdir):
    from scaler.core import find_safe_scale

    out = {"scales_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        history = cfg["history"]
        scale = cfg["initial"]
        for grads in history:
            want_scale, want_inf = ref.find_safe_scale(grads, scale)
            try:
                got_scale, got_inf = find_safe_scale(grads, scale)
            except Exception:
                got_scale, got_inf = -1.0, not want_inf
            if math_close(got_scale, want_scale) and bool(got_inf) == bool(want_inf):
                scale = want_scale
            else:
                break
        else:
            ok += 1
    out["scales_matched"] = float(ok)
    return out


def math_close(a, b):
    import math
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) < 1e-5
