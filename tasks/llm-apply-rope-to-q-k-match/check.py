import numpy as np

def _apply_rope_ref(x, pos):
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    assert d % 2 == 0
    omega = np.linspace(0.01, 0.99, d // 2)
    theta = pos * omega
    cos = np.cos(theta)
    sin = np.sin(theta)
    even = x[:, ::2]
    odd = x[:, 1::2]
    new_even = even * cos - odd * sin
    new_odd = even * sin + odd * cos
    out = np.empty_like(x)
    out[:, ::2] = new_even
    out[:, 1::2] = new_odd
    return out

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(0)
    n, d = 5, 8
    x = rng.standard_normal((n, d))
    pos = 3
    try:
        got = sol.apply_rope(x, pos)
    except Exception:
        return {"max_abs_err": float("inf")}
    ref = _apply_rope_ref(x, pos)
    if got.shape != ref.shape or got.dtype != np.float64:
        err = float("inf")
    else:
        err = float(np.max(np.abs(got - ref)))
    return {"max_abs_err": err}
