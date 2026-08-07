import ref
import numpy as np


def check(workdir):
    from export.normalize import normalize_to_scale_bias

    out = {"scale_bias_matched": 0.0}
    matched = 0
    total = len(ref.TEST_CONFIGS)

    for i, cfg in enumerate(ref.TEST_CONFIGS):
        want_scale, want_bias = ref.oracle_scale_bias(cfg["mean"], cfg["std"])
        try:
            got_scale, got_bias = normalize_to_scale_bias(cfg["mean"], cfg["std"])
        except Exception as e:
            out["_note"] = f"config {i} raised {type(e).__name__}: {e}"
            return out

        scale_ok = np.allclose(got_scale, want_scale, rtol=1e-5, atol=1e-7)
        bias_ok = np.allclose(got_bias, want_bias, rtol=1e-5, atol=1e-7)

        if scale_ok and bias_ok:
            matched += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got scale={got_scale}, bias={got_bias}; want scale={want_scale}, bias={want_bias}"

    if matched == total:
        out["scale_bias_matched"] = 1.0

    return out
