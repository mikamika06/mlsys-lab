import numpy as np

from mlsys import scorers


def _quant_rows_int4(V: np.ndarray) -> np.ndarray:
    """Symmetric INT4 round-to-nearest quantization, per row (output
    channel): scale by max(|row|)/7, round, clip to [-8, 7], rescale.
    """
    absmax = np.max(np.abs(V), axis=1, keepdims=True)
    absmax = np.where(absmax == 0, 1e-9, absmax)
    delta = absmax / 7.0
    return np.clip(np.round(V / delta), -8, 7) * delta


def _oracle(W: np.ndarray, X: np.ndarray):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Y_true = X @ W.T

    # Plain RTN: quantize W directly.
    W_hat_rtn = _quant_rows_int4(W)
    err_rtn = float(np.linalg.norm(X @ W_hat_rtn.T - Y_true) / np.linalg.norm(Y_true))

    # AWQ: per-input-channel scale = mean absolute activation on that
    # channel: scale W up before quantizing (protecting salient channels'
    # precision), quantize, then scale back down.
    s = np.mean(np.abs(X), axis=0)
    W_scaled = W * s[None, :]
    W_hat_scaled = _quant_rows_int4(W_scaled)
    W_hat_awq = W_hat_scaled / s[None, :]
    err_awq = float(np.linalg.norm(X @ W_hat_awq.T - Y_true) / np.linalg.norm(Y_true))

    reduction = 1.0 - err_awq / err_rtn
    return err_rtn, err_awq, reduction


def _synthetic_cases():
    rng = np.random.default_rng(24)
    cases = []
    for out_dim, in_dim, batch, salient, scale in [
        (10, 20, 40, (1, 5), 80.0),
        (8, 16, 32, (0, 3, 7), 150.0),
        (12, 24, 50, (10,), 60.0),
    ]:
        W = rng.normal(0.0, 1.0, size=(out_dim, in_dim))
        X = rng.normal(0.0, 1.0, size=(batch, in_dim))
        X[:, list(salient)] *= scale
        cases.append((W, X))
    return cases


def grade(sol, fx) -> dict:
    cases = [(fx["W"], fx["X"])] + _synthetic_cases()

    worst = 0.0
    for W, X in cases:
        ref = _oracle(W, X)
        try:
            got = sol.compare_awq_rtn_error(np.asarray(W).copy(), np.asarray(X).copy())
            got = tuple(float(v) for v in got)
        except Exception:
            return {"rel_err": float("inf")}

        if len(got) != 3 or not all(np.isfinite(v) for v in got):
            return {"rel_err": float("inf")}

        err = scorers.rel_err(np.array(ref), np.array(got))
        worst = max(worst, err)

    return {"rel_err": worst}
