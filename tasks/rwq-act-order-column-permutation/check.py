import numpy as np


def _col_scale_zp(w, nbits):
    qmax = (1 << nbits) - 1
    mn = min(0.0, float(np.min(w)))
    mx = max(0.0, float(np.max(w)))
    scale = (mx - mn) / qmax if mx > mn else 1.0
    zp = int(np.clip(round(-mn / scale), 0, qmax))
    return scale, zp


def _quant_val(w, scale, zp, nbits):
    qmax = (1 << nbits) - 1
    codes = np.clip(np.round(w / scale) + zp, 0, qmax)
    return (codes - zp) * scale


def _gptq_run(W, H, order, nbits, damp):
    W = np.asarray(W, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    d_out, d_in = W.shape
    order = np.asarray(order, dtype=np.int64)

    Hp = H[np.ix_(order, order)].copy()
    damp_val = damp * float(np.mean(np.diag(Hp)))
    Hp[np.diag_indices(d_in)] += damp_val
    Hinv = np.linalg.inv(Hp)

    scale_zp = [_col_scale_zp(W[:, c], nbits) for c in order]

    Wcur = W[:, order].copy()
    for i in range(d_in):
        w_col = Wcur[:, i]
        scale, zp = scale_zp[i]
        q_col = _quant_val(w_col, scale, zp, nbits)
        err = (w_col - q_col) / Hinv[i, i]
        Wcur[:, i] = q_col
        if i + 1 < d_in:
            Wcur[:, i + 1:] -= np.outer(err, Hinv[i, i + 1:])

    inv_order = np.argsort(order)
    Wq = Wcur[:, inv_order]
    mse = float(np.mean((Wq - W) ** 2))
    return Wq, mse


def grade(sol, fx) -> dict:
    """
    Builds random calibration Hessians H = X^T X and weight matrices W,
    computes the act-order permutation and runs the GPTQ-style sequential
    column quantization with a NumPy oracle; compares the submission's
    permutation (exact) and resulting MSE (relative error) to the oracle.
    """
    rng = np.random.default_rng(0)
    ok = 1.0
    worst_rel = 0.0
    for _ in range(6):
        d_in = int(rng.integers(4, 9))
        d_out = int(rng.integers(3, 7))
        n_cal = int(rng.integers(d_in + 3, d_in + 12))
        X = rng.normal(size=(n_cal, d_in))
        H = X.T @ X
        W = rng.normal(size=(d_out, d_in)) * rng.uniform(0.3, 2.0)
        nbits = int(rng.choice([2, 3, 4]))
        damp = 0.01

        perm_exp = np.argsort(-np.diag(H), kind="stable")
        _, mse_exp = _gptq_run(W, H, perm_exp, nbits, damp)

        try:
            perm_got, mse_got = sol.gptq_act_order(W.copy(), H.copy(), nbits, damp)
            perm_got = np.asarray(perm_got)
        except Exception:
            ok = 0.0
            worst_rel = float("inf")
            continue

        if perm_got.shape != perm_exp.shape or not np.array_equal(perm_got, perm_exp):
            ok = 0.0

        rel = abs(float(mse_got) - mse_exp) / (abs(mse_exp) + 1e-12)
        worst_rel = max(worst_rel, rel)

    return {"exact_match": ok, "rel_err": worst_rel}
