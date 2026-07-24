import numpy as np


def _affine_quant_dequant(x, bits, axis):
    """Uniform affine (asymmetric) min-max quantizer: scale/zero-point are
    derived from the min/max of each group along `axis` (axis=None means
    one group for the whole tensor)."""
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << bits) - 1
    xmin = np.min(x, axis=axis, keepdims=True)
    xmax = np.max(x, axis=axis, keepdims=True)
    scale = (xmax - xmin) / qmax
    scale = np.where(scale == 0, 1.0, scale)
    zero_point = np.round(-xmin / scale)
    q = np.clip(np.round(x / scale + zero_point), 0, qmax)
    return (q - zero_point) * scale


def _attention(K, V, q):
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    d = K.shape[1]
    logits = (K @ q) / np.sqrt(d)
    logits = logits - np.max(logits)
    w = np.exp(logits)
    w = w / np.sum(w)
    return w @ V


def _oracle(K, V, q, bits):
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    # KIVI scheme: keys quantized PER-CHANNEL (one scale/zero per column,
    # min/max taken across all tokens), values quantized PER-TOKEN (one
    # scale/zero per row, min/max taken across all channels).
    K_per_channel = _affine_quant_dequant(K, bits, axis=0)
    V_per_token = _affine_quant_dequant(V, bits, axis=1)

    # Baseline: keys quantized PER-TENSOR (one global scale/zero).
    K_per_tensor = _affine_quant_dequant(K, bits, axis=None)

    k_mse_per_channel = float(np.mean((K_per_channel - K) ** 2))
    k_mse_per_tensor = float(np.mean((K_per_tensor - K) ** 2))

    base = _attention(K, V, q)
    kivi_out = _attention(K_per_channel, V_per_token, q)
    attn_max_abs_err = float(np.max(np.abs(kivi_out - base)))

    return np.array([k_mse_per_channel, k_mse_per_tensor, attn_max_abs_err])


def _cases():
    rng = np.random.default_rng(0)
    cases = []
    for n, d, bits in [(16, 8, 4), (24, 12, 3), (32, 16, 4), (20, 10, 2)]:
        K = rng.standard_normal((n, d))
        V = rng.standard_normal((n, d))
        q = rng.standard_normal(d)
        cases.append((K, V, q, bits))
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for K, V, q, bits in _cases():
        ref = _oracle(K, V, q, bits)
        try:
            got = np.asarray(
                sol.kivi_quant_errors(K.copy(), V.copy(), q.copy(), bits),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
