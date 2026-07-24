import numpy as np


def _log_softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _scalar_loss(x, g):
    """The scalar whose gradient w.r.t. x IS the vjp we're checking:
    d/dx [ sum(g * log_softmax(x)) ] = log_softmax_vjp(x, g)."""
    return float(np.sum(g * _log_softmax(x)))


def _numeric_vjp(x, g, h=1e-5):
    """Central finite-difference oracle — perturbs each entry of x
    independently and re-evaluates the scalar loss above. Never touches
    the closed-form softmax formula the student is implementing."""
    dx = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        xp = x.copy(); xp[idx] += h
        xm = x.copy(); xm[idx] -= h
        dx[idx] = (_scalar_loss(xp, g) - _scalar_loss(xm, g)) / (2.0 * h)
    return dx


def grade(sol, fx) -> dict:
    rng = np.random.RandomState(21)
    shapes = [(4, 5), (3, 6), (1, 4), (5, 3)]
    max_err = 0.0

    for shape in shapes:
        x = rng.randn(*shape) * 2.0
        g = rng.randn(*shape)

        ref = _numeric_vjp(x, g)
        try:
            got = sol.log_softmax_vjp(x.copy(), g.copy())
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != x.shape:
            return {"max_abs_err": float("inf")}

        err = float(np.max(np.abs(got - ref)))
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
