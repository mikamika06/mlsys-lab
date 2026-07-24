"""Oracle: the TRUE gradient of a dropout-containing block, computed by
running its forward pass exactly once and reusing that one dropout mask for
backward (the behaviour a *correct* activation checkpoint must reproduce
when it recomputes the forward for backward). Analytic backprop through the
block is a closed form of two matmuls + relu + an elementwise mask, verified
independently against central finite differences below.
"""
import numpy as np


def _block(x, W1, W2, mask, p):
    h = x @ W1
    r = np.maximum(h, 0.0)
    d = r * mask / (1.0 - p)
    y = d @ W2
    return y, h, r, d


def _true_forward_and_grad(x, W1, W2, p, seed, n_pre, n_post, dY):
    """The reference: one shared RNG stream, `n_pre` unrelated draws before
    this block, this block's own mask draw, `n_post` unrelated draws after
    it -- then the gradient is computed by reusing THAT SAME mask (the only
    behaviour consistent with a faithfully-recomputed checkpoint)."""
    rng = np.random.default_rng(seed)
    for _ in range(n_pre):
        rng.random(3)
    mask = (rng.random((x.shape[0], W1.shape[1])) >= p).astype(np.float64)
    y, h, r, d = _block(x, W1, W2, mask, p)
    for _ in range(n_post):
        rng.random(3)

    dW2 = d.T @ dY
    dd = dY @ W2.T
    dr = dd * mask / (1.0 - p)
    dh = dr * (h > 0).astype(np.float64)
    dW1 = x.T @ dh
    dX = dh @ W1.T
    return y, dX, dW1, dW2


def _finite_diff_sanity():
    """One-time internal check that the analytic backprop above matches
    central finite differences, so the oracle itself is trustworthy."""
    rng = np.random.default_rng(123)
    n, m, k, o = 3, 2, 4, 2
    x = rng.standard_normal((n, m))
    W1 = rng.standard_normal((m, k))
    W2 = rng.standard_normal((k, o))
    dY = rng.standard_normal((n, o))
    p = 0.3
    mask = (rng.random((n, k)) >= p).astype(np.float64)

    def loss(x_, W1_, W2_):
        y_, _, _, _ = _block(x_, W1_, W2_, mask, p)
        return float(np.sum(y_ * dY))

    y, h, r, d = _block(x, W1, W2, mask, p)
    dW2 = d.T @ dY
    dd = dY @ W2.T
    dr = dd * mask / (1.0 - p)
    dh = dr * (h > 0).astype(np.float64)
    dW1 = x.T @ dh
    dX = dh @ W1.T

    eps = 1e-6
    worst = 0.0
    for arr, dref, name in ((x, dX, "x"), (W1, dW1, "W1"), (W2, dW2, "W2")):
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            xp, xm = x.copy(), x.copy()
            w1p, w1m = W1.copy(), W1.copy()
            w2p, w2m = W2.copy(), W2.copy()
            {"x": xp, "W1": w1p, "W2": w2p}[name][idx] += eps
            {"x": xm, "W1": w1m, "W2": w2m}[name][idx] -= eps
            fd = (loss(xp, w1p, w2p) - loss(xm, w1m, w2m)) / (2 * eps)
            worst = max(worst, abs(fd - dref[idx]))
    assert worst < 1e-6, f"oracle backprop disagrees with finite differences: {worst}"


_finite_diff_sanity()


def _cases():
    return [
        dict(seed=42, n_pre=2, n_post=3, p=0.4, data_seed=1),
        dict(seed=7, n_pre=0, n_post=5, p=0.5, data_seed=2),
        dict(seed=99, n_pre=4, n_post=1, p=0.25, data_seed=3),
    ]


def grade(sol, fx) -> dict:
    n, m, k, o = 4, 3, 5, 2
    worst = 0.0
    for c in _cases():
        drng = np.random.default_rng(c["data_seed"])
        x = drng.standard_normal((n, m))
        W1 = drng.standard_normal((m, k))
        W2 = drng.standard_normal((k, o))
        dY = drng.standard_normal((n, o))
        p = c["p"]

        y_ref, dX_ref, dW1_ref, dW2_ref = _true_forward_and_grad(
            x, W1, W2, p, c["seed"], c["n_pre"], c["n_post"], dY
        )

        try:
            out = sol.checkpointed_layer(
                x.copy(), W1.copy(), W2.copy(), p, c["seed"], c["n_pre"], c["n_post"], dY.copy()
            )
            y_got, dX_got, dW1_got, dW2_got = out
            y_got = np.asarray(y_got, dtype=np.float64)
            dX_got = np.asarray(dX_got, dtype=np.float64)
            dW1_got = np.asarray(dW1_got, dtype=np.float64)
            dW2_got = np.asarray(dW2_got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if (y_got.shape != y_ref.shape or dX_got.shape != dX_ref.shape
                or dW1_got.shape != dW1_ref.shape or dW2_got.shape != dW2_ref.shape):
            return {"max_abs_err": float("inf")}
        if not (np.all(np.isfinite(y_got)) and np.all(np.isfinite(dX_got))
                and np.all(np.isfinite(dW1_got)) and np.all(np.isfinite(dW2_got))):
            return {"max_abs_err": float("inf")}

        err = max(
            float(np.max(np.abs(y_got - y_ref))),
            float(np.max(np.abs(dX_got - dX_ref))),
            float(np.max(np.abs(dW1_got - dW1_ref))),
            float(np.max(np.abs(dW2_got - dW2_ref))),
        )
        worst = max(worst, err)

    return {"max_abs_err": worst}
