import numpy as np
from mlsys.scorers import max_abs_err

def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    seq_len = 7
    d_head = 5
    Q = rng.standard_normal((seq_len, d_head))
    K = rng.standard_normal((seq_len, d_head))
    V = rng.standard_normal((seq_len, d_head))

    scores = Q @ K.T / np.sqrt(d_head)
    ref_softmax = _softmax(scores)
    ref_out = ref_softmax @ V

    try:
        out = sol.sdpa_single_head(Q, K, V)
    except Exception:
        return {"max_abs_err": float("inf")}

    err = max_abs_err(ref_out, out)
    return {"max_abs_err": err}
