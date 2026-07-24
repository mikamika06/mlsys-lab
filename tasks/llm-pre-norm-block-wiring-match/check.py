import numpy as np
from mlsys import scorers


def _layer_norm(z, gamma, beta, eps=1e-5):
    """Reference LayerNorm over the last axis (population variance)."""
    mu = z.mean(axis=-1, keepdims=True)
    var = z.var(axis=-1, keepdims=True, ddof=0)
    return gamma * ((z - mu) / np.sqrt(var + eps)) + beta


def _softmax(s):
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def _attn(z, Wq, Wk, Wv, Wo):
    """Single-head scaled dot-product self-attention, no mask."""
    d = z.shape[-1]
    Q = z @ Wq
    K = z @ Wk
    V = z @ Wv
    scores = (Q @ K.T) / np.sqrt(d)
    A = _softmax(scores)
    return (A @ V) @ Wo


def _gelu(u):
    return 0.5 * u * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (u + 0.044715 * u ** 3)))


def _mlp(z, W1, b1, W2, b2):
    return _gelu(z @ W1 + b1) @ W2 + b2


def _ref_block(x, gamma1, beta1, gamma2, beta2, Wq, Wk, Wv, Wo, W1, b1, W2, b2):
    """Pure-NumPy pre-norm block: x + attn(LN1(x)), then h + mlp(LN2(h))."""
    h = x + _attn(_layer_norm(x, gamma1, beta1), Wq, Wk, Wv, Wo)
    y = h + _mlp(_layer_norm(h, gamma2, beta2), W1, b1, W2, b2)
    return y


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    # (T, d) cases; MLP hidden width h = 4*d
    cases = [(4, 8), (6, 16), (3, 32), (5, 12), (2, 24)]
    max_err = 0.0
    for T, d in cases:
        h = 4 * d
        x = rng.standard_normal((T, d))
        gamma1 = rng.standard_normal(d)
        beta1 = rng.standard_normal(d) * 0.5
        gamma2 = rng.standard_normal(d)
        beta2 = rng.standard_normal(d) * 0.5
        Wq = rng.standard_normal((d, d)) / np.sqrt(d)
        Wk = rng.standard_normal((d, d)) / np.sqrt(d)
        Wv = rng.standard_normal((d, d)) / np.sqrt(d)
        Wo = rng.standard_normal((d, d)) / np.sqrt(d)
        W1 = rng.standard_normal((d, h)) * 0.1
        b1 = rng.standard_normal(h) * 0.1
        W2 = rng.standard_normal((h, d)) * 0.1
        b2 = rng.standard_normal(d) * 0.1

        expected = _ref_block(x, gamma1, beta1, gamma2, beta2,
                              Wq, Wk, Wv, Wo, W1, b1, W2, b2)
        try:
            got = sol.pre_norm_block(x, gamma1, beta1, gamma2, beta2,
                                     Wq, Wk, Wv, Wo, W1, b1, W2, b2)
            got = np.asarray(got, dtype=np.float64)
            if got.shape != expected.shape:
                return {"max_abs_err": float("inf")}
        except Exception:
            return {"max_abs_err": float("inf")}

        err = scorers.max_abs_err(expected, got)
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
