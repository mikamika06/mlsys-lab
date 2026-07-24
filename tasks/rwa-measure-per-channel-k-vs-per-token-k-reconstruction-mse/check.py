import numpy as np


def _affine_quant_dequant(x, bits, axis):
    """Uniform affine (asymmetric) min-max quantizer/dequantizer. The
    min/max (hence scale/zero-point) are computed per-group along `axis`;
    every element of a group shares that one scale/zero-point."""
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << bits) - 1
    xmin = np.min(x, axis=axis, keepdims=True)
    xmax = np.max(x, axis=axis, keepdims=True)
    scale = (xmax - xmin) / qmax
    scale = np.where(scale == 0, 1.0, scale)
    zero_point = np.round(-xmin / scale)
    q = np.clip(np.round(x / scale + zero_point), 0, qmax)
    return (q - zero_point) * scale


def _oracle(K, bits):
    K = np.asarray(K, dtype=np.float64)
    # per-channel: one scale/zero PER COLUMN, min/max taken over all tokens (axis=0)
    K_per_channel = _affine_quant_dequant(K, bits, axis=0)
    # per-token: one scale/zero PER ROW, min/max taken over all channels (axis=1)
    K_per_token = _affine_quant_dequant(K, bits, axis=1)
    mse_per_channel = float(np.mean((K_per_channel - K) ** 2))
    mse_per_token = float(np.mean((K_per_token - K) ** 2))
    return np.array([mse_per_channel, mse_per_token])


def _cases():
    rng = np.random.default_rng(0)
    cases = []
    for n_tokens, d, bits, bias_lo, bias_hi, noise_std in [
        (16, 8, 4, -8.0, 8.0, 0.15),
        (24, 12, 3, -5.0, 5.0, 0.1),
        (32, 16, 4, -10.0, 10.0, 0.2),
        (20, 10, 2, -6.0, 6.0, 0.1),
        (40, 20, 4, -12.0, 12.0, 0.25),
        (12, 6, 4, -4.0, 4.0, 0.05),
    ]:
        # Each channel (RoPE-rotated key dimension) carries its own roughly
        # constant bias across tokens, plus small per-token noise -- the
        # structure that makes keys favor per-channel grouping.
        channel_bias = rng.uniform(bias_lo, bias_hi, size=d)
        noise = rng.normal(0.0, noise_std, size=(n_tokens, d))
        K = channel_bias[None, :] + noise
        cases.append((K, bits))
    return cases


def grade(sol, fx) -> dict:
    worst_rel = 0.0
    for K, bits in _cases():
        ref = _oracle(K, bits)
        try:
            got = np.asarray(
                sol.per_channel_vs_per_token_k_mse(K.copy(), bits),
                dtype=np.float64,
            )
        except Exception:
            return {"mse": float("inf")}
        if got.shape != ref.shape:
            return {"mse": float("inf")}
        # KIVI's core finding must actually hold: per-channel K MSE below
        # per-token K MSE. A solution that reports two numbers close to
        # the oracle but in the wrong order still fails.
        if not (float(got[0]) < float(got[1])):
            return {"mse": float("inf")}
        rel = np.abs(got - ref) / (np.abs(ref) + 1e-12)
        worst_rel = max(worst_rel, float(np.max(rel)))
    return {"mse": worst_rel}
