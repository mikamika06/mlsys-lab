import numpy as np


def _quantize_group_int4(W, group_size):
    W = np.asarray(W, dtype=np.float64)
    out = np.empty_like(W)
    for start in range(0, W.shape[1], group_size):
        end = min(start + group_size, W.shape[1])
        group = W[:, start:end]
        scale = max(float(np.max(np.abs(group))) / 7.0, 1e-12)
        out[:, start:end] = np.clip(np.round(group / scale), -8, 7) * scale
    return out


def _oracle(W, X, group_size):
    plain = _quantize_group_int4(W, group_size)

    importance = np.mean(np.abs(X), axis=0)
    channel_scale = (importance / (np.mean(importance) + 1e-12)) ** 0.5

    scaled = W * channel_scale
    awq = _quantize_group_int4(scaled, group_size) / channel_scale

    y = X @ W.T
    plain_mse = float(np.mean((y - X @ plain.T) ** 2))
    awq_mse = float(np.mean((y - X @ awq.T) ** 2))
    return awq_mse, plain_mse


def grade(sol, fx) -> dict:
    cases = []
    for seed in [0, 2, 4, 7]:
        rng = np.random.default_rng(seed)
        W = rng.normal(size=(16, 32)) * np.exp(rng.normal(0, 0.8, size=(1, 32)))
        X = rng.normal(size=(128, 32))
        X[:, :4] *= 10.0
        cases.append((W, X, 8))

    awq_err = 0.0
    plain_err = 0.0
    beats = 1.0

    for W, X, group_size in cases:
        try:
            got_awq, got_plain = sol.awq_vs_plain_group_int4_mse(W, X, group_size)
        except Exception:
            return {
                "awq_mse_error": 1.0,
                "plain_mse_error": 1.0,
                "awq_beats_plain": 0.0
            }

        ref_awq, ref_plain = _oracle(W, X, group_size)
        awq_err = max(awq_err, abs(float(got_awq) - ref_awq))
        plain_err = max(plain_err, abs(float(got_plain) - ref_plain))

        if not (ref_awq < ref_plain):
            beats = 0.0

    return {
        "awq_mse_error": awq_err,
        "plain_mse_error": plain_err,
        "awq_beats_plain": beats
    }
