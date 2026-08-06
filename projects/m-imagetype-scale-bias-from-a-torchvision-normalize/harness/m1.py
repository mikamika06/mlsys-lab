import ref
import numpy as np

def check(workdir):
    from imageconv.scale import get_scale_bias
    out = {"configs_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want_s, want_b = ref.get_scale_bias(cfg["mean"], cfg["std"])
        try:
            got_s, got_b = get_scale_bias(cfg["mean"], cfg["std"])
            if np.allclose(got_s, want_s) and np.allclose(got_b, want_b):
                ok += 1
        except Exception:
            pass
    out["configs_matched"] = float(ok)
    return out
