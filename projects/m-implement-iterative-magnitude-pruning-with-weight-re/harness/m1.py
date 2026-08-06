import ref
import numpy as np


def check(workdir):
    from prune.imp import iterative_prune
    np.random.seed(42)
    init_weights = [np.random.randn(10, 10), np.random.randn(10)]
    data = [np.random.randn(10, 10), np.random.randn(10)]

    want_w, want_m = ref.run_reference_imp(init_weights, data)
    try:
        got_w, got_m = iterative_prune(ref.dummy_model_fn, data, 3, 0.5, init_weights)
    except Exception as e:
        return {"imp_matched": 0.0, "_note": f"raised exception: {e}"}

    w_match = all(np.allclose(gw, ww, atol=1e-5) for gw, ww in zip(got_w, want_w))
    m_match = all(np.allclose(gm, wm, atol=1e-5) for gm, wm in zip(got_m, want_m))

    match = 1.0 if (w_match and m_match) else 0.0
    out = {"imp_matched": match}
    if match == 0.0:
        out["_note"] = "iterative prune outputs do not match reference"
    return out
