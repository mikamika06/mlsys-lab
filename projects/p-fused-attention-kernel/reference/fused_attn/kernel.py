import numpy as np


def online_softmax_step(chunk, m_prev, d_prev):
    m_chunk = np.max(chunk, axis=-1, keepdims=True)
    m_new = np.maximum(m_prev, m_chunk)
    exp_prev = np.exp(m_prev - m_new)
    exp_chunk = np.exp(chunk - m_new)
    d_new = exp_prev * d_prev + np.sum(exp_chunk, axis=-1, keepdims=True)
    return m_new, d_new, exp_chunk


def tiled_attention(q, k, v, block_size=16, causal=False):
    seq_len, dim = q.shape
    scale = 1.0 / np.sqrt(dim)
    o = np.zeros_like(q)
    m = np.full((seq_len, 1), -np.inf)
    d = np.zeros((seq_len, 1))
    num_blocks = (seq_len + block_size - 1) // block_size
    for j in range(num_blocks):
        start_j = j * block_size
        end_j = min(start_j + block_size, seq_len)
        k_block = k[start_j:end_j]
        v_block = v[start_j:end_j]
        scores = np.matmul(q, k_block.T) * scale
        if causal:
            row_idx = np.arange(seq_len)[:, None]
            col_idx = np.arange(start_j, end_j)[None, :]
            mask = col_idx > row_idx
            scores = np.where(mask, -1e9, scores)
        for i in range(seq_len):
            row_scores = scores[i:i+1]
            m_prev = m[i:i+1]
            d_prev = d[i:i+1]
            m_new, d_new, exp_chunk = online_softmax_step(row_scores, m_prev, d_prev)
            acc_scale = np.exp(m_prev - m_new)
            o[i:i+1] = o[i:i+1] * acc_scale + np.matmul(exp_chunk, v_block)
            m[i:i+1] = m_new
            d[i:i+1] = d_new
    o = o / d
    return o
