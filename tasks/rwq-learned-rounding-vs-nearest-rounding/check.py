import itertools

import numpy as np

D_OUT, D_IN, N_CAL, NBITS = 4, 6, 30, 3


def _row_scale(w, qmax):
    a = float(np.max(np.abs(w)))
    return a / qmax if a > 0 else 1.0


def _oracle(W, X, nbits):
    """Per row: brute-force the best floor/ceil rounding-direction choice
    per column (2**d_in candidates) to minimize the row's output SSE
    (a tractable stand-in for AutoRound/AdaRound-style learned rounding),
    versus plain nearest-rounding (RTN)."""
    d_out, d_in = W.shape
    qmax = (1 << (nbits - 1)) - 1
    combos = np.array(list(itertools.product([0, 1], repeat=d_in)))  # (2**d_in, d_in)

    total_learned = 0.0
    total_rtn = 0.0
    for i in range(d_out):
        w = W[i]
        s = _row_scale(w, qmax)
        y = w @ X.T

        f = np.clip(np.floor(w / s), -qmax, qmax)
        c = np.clip(np.ceil(w / s), -qmax, qmax)
        cand = np.stack([f, c], axis=1)  # (d_in, 2)
        chosen = cand[np.arange(d_in)[None, :], combos]  # (2**d_in, d_in)
        v = chosen * s
        Yhat = v @ X.T  # (2**d_in, n_cal)
        sse = np.sum((Yhat - y[None, :]) ** 2, axis=1)
        total_learned += float(np.min(sse))

        code_rtn = np.clip(np.round(w / s), -qmax, qmax)
        v_rtn = code_rtn * s
        total_rtn += float(np.sum((v_rtn @ X.T - y) ** 2))

    n = d_out * X.shape[0]
    return total_learned / n, total_rtn / n


def grade(sol, fx) -> dict:
    """
    Builds several seeded random (W, X) layers and, for each, computes the
    brute-force-optimal "learned rounding" output MSE and the plain
    nearest-rounding (RTN) output MSE with a NumPy oracle at fixed bit
    width. Compares the submission's two reported MSEs (relative error)
    to the oracle's, and checks that the submission's learned MSE never
    exceeds its own RTN MSE.
    """
    rng = np.random.default_rng(0)
    learned_rel_worst = 0.0
    rtn_rel_worst = 0.0
    le_ok = 1.0
    for _ in range(5):
        X = rng.normal(size=(N_CAL, D_IN))
        W = rng.normal(size=(D_OUT, D_IN)) * rng.uniform(0.4, 2.0, size=(1, D_IN))

        mse_learned_exp, mse_rtn_exp = _oracle(W, X, NBITS)

        try:
            mse_learned_got, mse_rtn_got = sol.rounding_output_mse(W.copy(), X.copy(), NBITS)
            mse_learned_got = float(mse_learned_got)
            mse_rtn_got = float(mse_rtn_got)
        except Exception:
            return {"learned_rel_err": float("inf"), "rtn_rel_err": float("inf"), "learned_le_rtn": 0.0}

        learned_rel = abs(mse_learned_got - mse_learned_exp) / (abs(mse_learned_exp) + 1e-12)
        rtn_rel = abs(mse_rtn_got - mse_rtn_exp) / (abs(mse_rtn_exp) + 1e-12)
        learned_rel_worst = max(learned_rel_worst, learned_rel)
        rtn_rel_worst = max(rtn_rel_worst, rtn_rel)
        if mse_learned_got > mse_rtn_got + 1e-9:
            le_ok = 0.0

    return {
        "learned_rel_err": learned_rel_worst,
        "rtn_rel_err": rtn_rel_worst,
        "learned_le_rtn": le_ok,
    }
