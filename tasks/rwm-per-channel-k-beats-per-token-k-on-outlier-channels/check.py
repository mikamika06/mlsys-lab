import numpy as np
from mlsys.scorers import rel_err


def _quantize_symmetric(K, bits, group_axis):
    qmax = 2 ** (bits - 1) - 1
    amax = np.max(np.abs(K), axis=group_axis, keepdims=True)
    scale = np.where(amax > 1e-12, amax / qmax, 1.0)
    q = np.clip(np.round(K / scale), -qmax, qmax)
    return q * scale


def _oracle(K, bits):
    deq_channel = _quantize_symmetric(K, bits, group_axis=0)
    deq_token = _quantize_symmetric(K, bits, group_axis=1)
    mse_channel = float(np.mean((K - deq_channel) ** 2))
    mse_token = float(np.mean((K - deq_token) ** 2))
    return mse_channel, mse_token


def _cases():
    rng = np.random.default_rng(0)
    cases = []

    for n_tokens, d_channels, bits, n_outliers, outlier_std in [
        (48, 16, 4, 3, 25.0),
        (64, 24, 3, 4, 40.0),
        (32, 12, 8, 2, 15.0),
    ]:
        K = rng.standard_normal((n_tokens, d_channels)) * 0.3
        outlier_cols = rng.choice(d_channels, size=n_outliers, replace=False)
        for c in outlier_cols:
            K[:, c] = rng.standard_normal(n_tokens) * outlier_std
        cases.append((K, bits))

    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for K, bits in _cases():
        ref = np.array(_oracle(K.astype(np.float64), bits), dtype=np.float64)

        try:
            got = sol.compare_k_quant_granularity(K.copy(), bits)
            got_mc, got_mt = got
            got_arr = np.array([float(got_mc), float(got_mt)], dtype=np.float64)
        except Exception:
            return {"rel_err": 1.0}

        err = rel_err(ref, got_arr)

        # The whole point of the task: per-channel must beat per-token by a
        # wide margin on outlier-channel data. If the oracle shows a clear
        # gap but the submission's own numbers don't reflect it, fail hard
        # even if the raw rel_err happened to look small.
        if ref[1] > ref[0] * 2.0 and not (got_arr[0] < got_arr[1]):
            err = max(err, 1.0)

        worst = max(worst, err)

    return {"rel_err": worst}
