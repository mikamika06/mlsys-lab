import ref
import numpy as np


def check(workdir):
    from aneattn import layout

    out = {"layouts_matched": 0.0}
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, 4, 16, 32))

    try:
        got = layout.to_ane_friendly(x)
        want = ref.get_ane_output(x)
        if got.shape == (1, 128, 1, 16) and np.allclose(got.shape, want.shape):
            out["layouts_matched"] = 1.0
        else:
            out["_note"] = f"shape mismatch: got {got.shape}, want (1, 128, 1, 16)"
    except Exception as e:
        out["_note"] = f"exception in layout: {str(e)[:100]}"
    return out
