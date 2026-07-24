import numpy as np
from mlsys.scorers import max_abs_err

def _ref_rope(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Reference RoPE implementation using rotation matrices.
    """
    x = np.asarray(x, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    batch, seq_len, dim = x.shape
    assert dim % 2 == 0, "Dimension must be even"
    freqs = 10000 ** (-np.arange(0, dim // 2) / (dim / 2))
    theta = pos[:, None] * freqs[None, :]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    new_even = x_even * cos_t - x_odd * sin_t
    new_odd = x_even * sin_t + x_odd * cos_t

    out = np.empty_like(x)
    out[..., ::2] = new_even
    out[..., 1::2] = new_odd
    return out

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        {"batch": 1, "seq_len": 5, "dim": 8},
        {"batch": 2, "seq_len": 10, "dim": 12},
        {"batch": 3, "seq_len": 7, "dim": 16},
        {"batch": 4, "seq_len": 6, "dim": 20},
    ]
    max_error = 0.0
    for case in cases:
        batch = case["batch"]
        seq_len = case["seq_len"]
        dim = case["dim"]
        x = rng.standard_normal((batch, seq_len, dim))
        pos = np.arange(seq_len).astype(np.float64)
        try:
            cand = sol.rope_complex(x, pos)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _ref_rope(x, pos)
        err = max_abs_err(ref, cand)
        if err > max_error:
            max_error = err
    return {"max_abs_err": max_error}
