import numpy as np


def _oracle_gptq(W: np.ndarray, Hinv: np.ndarray, scales: np.ndarray, bits: int) -> dict:
    r, d = W.shape
    L = 2 ** (bits - 1) - 1

    Wc = W.copy()
    codes = np.zeros((r, d), dtype=np.int64)

    for q in range(d):
        w_q = Wc[:, q]
        c_q = np.clip(np.round(w_q / scales), -L, L)
        codes[:, q] = c_q.astype(np.int64)
        deq = c_q * scales
        err = w_q - deq
        Wc[:, q] = deq

        if q + 1 < d:
            factor = err / Hinv[q, q]
            Wc[:, q + 1:] -= np.outer(factor, Hinv[q, q + 1:])

    return {"codes": codes, "W_hat": Wc}


def _build_case():
    rng = np.random.default_rng(0)
    r, d = 8, 16
    W = (rng.standard_normal((r, d)) * 0.05).astype(np.float64)
    bits = 4
    L = 2 ** (bits - 1) - 1
    scales = np.max(np.abs(W), axis=1) / L
    scales = np.where(scales == 0, 1.0, scales)

    n_samples = 64
    X = rng.standard_normal((n_samples, d))
    H = (X.T @ X) / n_samples + 1e-2 * np.eye(d)
    Hinv = np.linalg.inv(H)

    return W, Hinv, scales, bits


def grade(sol, fx) -> dict:
    W, Hinv, scales, bits = _build_case()
    oracle = _oracle_gptq(W, Hinv, scales, bits)
    oracle_mse = float(np.mean((W - oracle["W_hat"]) ** 2))

    try:
        got = sol.gptq_quantize(np.array(W, dtype=np.float64), np.array(Hinv, dtype=np.float64),
                                 np.array(scales, dtype=np.float64), bits)
        codes = np.asarray(got["codes"])
        W_hat = np.asarray(got["W_hat"], dtype=np.float64)
        if codes.shape != oracle["codes"].shape or W_hat.shape != W.shape:
            return {"codes_max_abs_err": float("inf"), "mse_abs_diff": float("inf")}
    except Exception:
        return {"codes_max_abs_err": float("inf"), "mse_abs_diff": float("inf")}

    codes_max_abs_err = float(np.max(np.abs(codes.astype(np.int64) - oracle["codes"])))
    got_mse = float(np.mean((W - W_hat) ** 2))
    mse_abs_diff = abs(got_mse - oracle_mse)

    return {"codes_max_abs_err": codes_max_abs_err, "mse_abs_diff": mse_abs_diff}
