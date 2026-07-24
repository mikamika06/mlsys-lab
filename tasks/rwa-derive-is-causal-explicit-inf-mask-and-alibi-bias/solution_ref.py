import numpy as np


def apply_attention_bias(logits, is_causal=False, alibi_slope=None):
    out = np.asarray(logits, dtype=np.float64).copy()
    q, k = out.shape

    if alibi_slope is not None:
        q_idx = np.arange(q)[:, None]
        kv_idx = np.arange(k)[None, :]
        out += float(alibi_slope) * (kv_idx - q_idx)

    if is_causal:
        mask = np.zeros((q, k), dtype=np.float64)
        mask[np.triu_indices(q, k, 1)] = -np.inf
        out += mask

    return out
