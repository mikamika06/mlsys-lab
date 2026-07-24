import numpy as np


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def mha_gqa_mqa_reconstruct(Q: np.ndarray, K: np.ndarray, V: np.ndarray, group_sizes):
    """
    Reference: same input Q/K/V through several KV-grouping arities.

    For each group_size g in group_sizes, pool K and V within each group of
    g adjacent heads (mean), broadcast the pooled K/V back to n_heads heads,
    and run standard scaled dot-product attention with the original Q.
    g == 1 reproduces exact MHA; g == n_heads is MQA; anything in between is
    GQA(g).
    """
    batch, seq_q, n_heads, d = Q.shape
    seq_k = K.shape[1]

    results = []
    for g in group_sizes:
        n_kv = n_heads // g

        Kg = K.reshape(batch, seq_k, n_kv, g, d).mean(axis=3)
        Vg = V.reshape(batch, seq_k, n_kv, g, d).mean(axis=3)
        K_bc = np.repeat(Kg, g, axis=2)
        V_bc = np.repeat(Vg, g, axis=2)

        Qh = Q.transpose(0, 2, 1, 3)
        Kh = K_bc.transpose(0, 2, 1, 3)
        Vh = V_bc.transpose(0, 2, 1, 3)

        scores = (Qh @ Kh.swapaxes(-2, -1)) / np.sqrt(d)
        weights = _softmax(scores, axis=-1)
        out = (weights @ Vh).transpose(0, 2, 1, 3)

        size_ratio = n_kv / n_heads
        results.append((out, size_ratio))

    return results
