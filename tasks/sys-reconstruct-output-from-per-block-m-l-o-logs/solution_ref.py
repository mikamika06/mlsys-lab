import math
import numpy as np


def reconstruct_attention_from_block_logs(block_m: np.ndarray, block_l: np.ndarray,
                                           block_o: np.ndarray) -> np.ndarray:
    """Reconstruct the single global, correctly-normalized attention output
    vector from K blocks' LOCAL online-softmax summaries -- no raw scores
    or values are available, only:

      block_m: (K,)   each block's local max score.
      block_l: (K,)   each block's local softmax denominator
                       sum(exp(score - block_m[k])) over that block only.
      block_o: (K, d) each block's local weighted value accumulator
                       sum(exp(score - block_m[k])[:, None] * value, axis=0)
                       over that block only.

    Combine sequentially with the online-softmax merge rule: for a running
    (m, l, o) and a new block (m_k, l_k, o_k),

        m_new = max(m, m_k)
        l_new = l * exp(m - m_new) + l_k * exp(m_k - m_new)
        o_new = o * exp(m - m_new) + o_k * exp(m_k - m_new)

    which is exact (associative/commutative) regardless of merge order.
    The final output is o_final / l_final.
    """
    block_m = np.asarray(block_m, dtype=np.float64)
    block_l = np.asarray(block_l, dtype=np.float64)
    block_o = np.asarray(block_o, dtype=np.float64)

    K = block_m.shape[0]
    m = block_m[0]
    l = block_l[0]
    
    o_src = block_o[0]
    d = o_src.shape[0]
    o = np.empty(d, dtype=np.float64)
    for j in range(d):
        o[j] = o_src[j]

    for k in range(1, K):
        mk = block_m[k]
        if m > mk:
            m_new = m
        else:
            m_new = mk

        scale_old = math.exp(m - m_new)
        scale_new = math.exp(mk - m_new)

        l = l * scale_old + block_l[k] * scale_new

        bk_o = block_o[k]
        for j in range(d):
            o[j] = o[j] * scale_old + bk_o[j] * scale_new

        m = m_new

    result = np.empty(d, dtype=np.float64)
    for j in range(d):
        result[j] = o[j] / l

    return result
