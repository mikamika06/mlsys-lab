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
    o = block_o[0].copy()

    for k in range(1, K):
        m_new = max(m, block_m[k])
        scale_old = np.exp(m - m_new)
        scale_new = np.exp(block_m[k] - m_new)
        l = l * scale_old + block_l[k] * scale_new
        o = o * scale_old + block_o[k] * scale_new
        m = m_new

    return o / l
