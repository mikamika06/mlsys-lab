import numpy as np

BITS = 4
DAMP = 0.01


def _row_grid(W, bits):
    qmax = 2 ** (bits - 1) - 1
    amax = np.max(np.abs(W), axis=1)
    amax = np.where(amax == 0, 1.0, amax)
    scale = amax / qmax
    return scale, qmax


def _rtn(W, bits):
    scale, qmax = _row_grid(W, bits)
    return np.clip(np.round(W / scale[:, None]), -qmax, qmax) * scale[:, None]


def _gptq(W, X, bits, damp=DAMP):
    n = X.shape[0]
    d_out, d_in = W.shape
    H = (X.T @ X) / n
    H = H + damp * np.mean(np.diag(H)) * np.eye(d_in)
    Hinv = np.linalg.inv(H)
    scale, qmax = _row_grid(W, bits)

    Wq = np.zeros_like(W)
    Werr = W.copy()
    Hinv = Hinv.copy()
    for j in range(d_in):
        d = Hinv[j, j]
        w = Werr[:, j]
        q = np.clip(np.round(w / scale), -qmax, qmax) * scale
        Wq[:, j] = q
        err = (w - q) / d
        if j + 1 < d_in:
            Werr[:, j + 1:] -= np.outer(err, Hinv[j, j + 1:])
            Hinv[j + 1:, j + 1:] -= np.outer(Hinv[j + 1:, j], Hinv[j, j + 1:]) / d
    return Wq


def _mse(X, W, Wq):
    Y = X @ W.T
    Yq = X @ Wq.T
    return float(np.mean((Y - Yq) ** 2))


def _make_problem(rng, n=64, d_out=6, d_in=8):
    k = 4
    Z = rng.standard_normal((n, k))
    A = rng.standard_normal((k, d_in))
    X = Z @ A + 0.1 * rng.standard_normal((n, d_in))
    W = rng.standard_normal((d_out, d_in))
    return X, W


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    trials = 5

    rtn_err = 0.0
    gptq_err = 0.0
    ref_rtn_total = 0.0
    ref_gptq_total = 0.0
    got_rtn_total = 0.0
    got_gptq_total = 0.0

    for _ in range(trials):
        X, W = _make_problem(rng)
        Wq_rtn = _rtn(W, BITS)
        Wq_gptq = _gptq(W, X, BITS)
        ref_rtn = _mse(X, W, Wq_rtn)
        ref_gptq = _mse(X, W, Wq_gptq)

        try:
            got_rtn, got_gptq = sol.gptq_vs_rtn_output_error(W.copy(), X.copy(), BITS)
            got_rtn = float(got_rtn)
            got_gptq = float(got_gptq)
        except Exception:
            return {
                "rtn_mse_error": float("inf"),
                "gptq_mse_error": float("inf"),
                "gptq_beats_rtn": 0.0,
            }

        rtn_err = max(rtn_err, abs(got_rtn - ref_rtn))
        gptq_err = max(gptq_err, abs(got_gptq - ref_gptq))
        ref_rtn_total += ref_rtn
        ref_gptq_total += ref_gptq
        got_rtn_total += got_rtn
        got_gptq_total += got_gptq

    return {
        "rtn_mse_error": rtn_err,
        "gptq_mse_error": gptq_err,
        "gptq_beats_rtn": float(got_gptq_total < got_rtn_total),
    }
