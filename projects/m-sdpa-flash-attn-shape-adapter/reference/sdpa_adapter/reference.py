import numpy as np

def reference_attention(q, k, v, is_causal=False, scale=None):
    if scale is None:
        scale = 1.0 / np.sqrt(q.shape[-1])

    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    seq_q = q.shape[-2]
    seq_k = k.shape[-2]

    if is_causal:
        i_indices = np.arange(seq_q)[:, None]
        j_indices = np.arange(seq_k)[None, :]
        mask = j_indices > (seq_k - seq_q + i_indices)
        scores = np.where(mask, -1e9, scores)

    row_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - row_max)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)

    lse = np.squeeze(row_max, axis=-1) + np.log(np.squeeze(sum_exp, axis=-1) + 1e-12)
    attention_weights = exp_scores / (sum_exp + 1e-12)
    output = np.matmul(attention_weights, v)
    return output, lse
