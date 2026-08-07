import numpy as np


class MemoryTracker:

    def __init__(self):
        self.reads = 0
        self.writes = 0

    def reset(self):
        self.reads = 0
        self.writes = 0

    def track_read(self, size_elements):
        self.reads += int(size_elements)

    def track_write(self, size_elements):
        self.writes += int(size_elements)


def online_softmax_update(m_prev, l_prev, acc_prev, S_block, V_block):
    m_curr = np.maximum(m_prev, np.max(S_block, axis=-1, keepdims=True))
    P_block = np.exp(S_block - m_curr)
    alpha = np.exp(m_prev - m_curr)
    l_curr = alpha * l_prev + np.sum(P_block, axis=-1, keepdims=True)
    acc_curr = alpha * acc_prev + np.matmul(P_block, V_block)
    return m_curr, l_curr, acc_curr


def naive_attention(Q, K, V, sm_scale=1.0, is_causal=False):
    S = np.matmul(Q, K.swapaxes(-1, -2)) * sm_scale
    if is_causal:
        seq_len_q = Q.shape[-2]
        seq_len_k = K.shape[-2]
        mask = np.triu(np.ones((seq_len_q, seq_len_k), dtype=bool), k=1)
        S = np.where(mask, -1e9, S)
    m = np.max(S, axis=-1, keepdims=True)
    P = np.exp(S - m)
    P = P / np.sum(P, axis=-1, keepdims=True)
    O = np.matmul(P, V)
    return O
