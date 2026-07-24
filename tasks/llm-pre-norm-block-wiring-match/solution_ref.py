import numpy as np


def _layer_norm(z, gamma, beta, eps=1e-5):
    mu = z.mean(axis=-1, keepdims=True)
    var = z.var(axis=-1, keepdims=True, ddof=0)
    return gamma * ((z - mu) / np.sqrt(var + eps)) + beta


def _softmax(s):
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def _attn(z, Wq, Wk, Wv, Wo):
    d = z.shape[-1]
    Q = z @ Wq
    K = z @ Wk
    V = z @ Wv
    scores = (Q @ K.T) / np.sqrt(d)
    return (_softmax(scores) @ V) @ Wo


def _gelu(u):
    return 0.5 * u * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (u + 0.044715 * u ** 3)))


def _mlp(z, W1, b1, W2, b2):
    return _gelu(z @ W1 + b1) @ W2 + b2


def pre_norm_block(x, gamma1, beta1, gamma2, beta2,
                   Wq, Wk, Wv, Wo, W1, b1, W2, b2):
    """One pre-norm transformer block over a residual stream x of shape (T, d).

    Attention sublayer:  h = x + attn(LN1(x))
    MLP sublayer:        y = h + mlp(LN2(h))
    """
    x = np.asarray(x, dtype=np.float64)
    # Attention sublayer: normalize the input, add back onto the raw stream.
    h = x + _attn(_layer_norm(x, gamma1, beta1), Wq, Wk, Wv, Wo)
    # MLP sublayer: second norm sees h (not x), residual add onto h.
    y = h + _mlp(_layer_norm(h, gamma2, beta2), W1, b1, W2, b2)
    return y
