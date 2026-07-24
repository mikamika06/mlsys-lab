import numpy as np


def _quant(W, scale, r, qmin, qmax):
    q = np.clip(np.floor(W / scale + r + 0.5), qmin, qmax)
    return q * scale


def _rtn(W, bits):
    W = np.asarray(W, dtype=np.float64)
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    scale = np.max(np.abs(W)) / qmax
    return _quant(W, scale, 0.0, qmin, qmax)


def _oracle(W, bits, steps, lr, seed):
    np.random.seed(seed)
    W = np.asarray(W, dtype=np.float64)

    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    scale = np.max(np.abs(W)) / qmax

    r = np.zeros_like(W, dtype=np.float64)

    best = _quant(W, scale, r, qmin, qmax)
    best_mse = float(np.mean((best - W) ** 2))

    for _ in range(steps):
        current = _quant(W, scale, r, qmin, qmax)
        current_mse = float(np.mean((current - W) ** 2))
        if current_mse < best_mse:
            best = current
            best_mse = current_mse

        grad = (2.0 * scale / W.size) * (current - W)
        r = r - lr * np.sign(grad)

    current = _quant(W, scale, r, qmin, qmax)
    current_mse = float(np.mean((current - W) ** 2))
    if current_mse < best_mse:
        best = current
        best_mse = current_mse

    return best.astype(np.float64), best_mse


def grade(sol, fx) -> dict:
    cases = [
        (np.array([[0.2, -0.8], [1.1, -1.5]], dtype=np.float64), 3, 10, 0.05, 0),
        (np.array([[0.31, 0.72, -1.2], [1.9, -0.4, 0.05]], dtype=np.float64), 4, 15, 0.03, 7),
        (np.array([[2.0, -1.0, 0.0], [0.6, -0.7, 1.4]], dtype=np.float64), 3, 20, 0.04, 2),
    ]

    mse_metric = 0.0
    better_metric = 1.0

    for W, bits, steps, lr, seed in cases:
        ref_wq, ref_mse = _oracle(W, bits, steps, lr, seed)
        rtn = _rtn(W, bits)
        rtn_mse = float(np.mean((rtn - W) ** 2))

        try:
            got_wq, got_mse = sol.autoround_block(W, bits, steps, lr, seed)
        except Exception:
            return {"mse": 1e9, "not_worse_than_rtn": 0.0}

        if not np.array_equal(np.asarray(got_wq, dtype=np.float64), ref_wq):
            mse_metric = 1e9
        else:
            mse_metric = max(mse_metric, abs(float(got_mse) - ref_mse))

        if float(got_mse) > rtn_mse + 1e-12:
            better_metric = 0.0

    return {
        "mse": mse_metric,
        "not_worse_than_rtn": better_metric
    }
