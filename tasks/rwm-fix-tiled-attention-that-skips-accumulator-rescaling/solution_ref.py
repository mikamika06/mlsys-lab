import numpy as np


def tiled_attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, block_size: int) -> np.ndarray:
    """
    Non-causal full attention O = softmax(Q K^T) V, computed by streaming
    K/V in blocks of `block_size` rows while maintaining a per-query running
    max `m`, running denominator `l`, and running numerator accumulator `acc`
    (the standard online-softmax / FlashAttention forward recurrence).

    Q : (n_q, d)
    K : (n_k, d)
    V : (n_k, d_v)
    returns O : (n_q, d_v), float64
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n_q = Q.shape[0]
    d_v = V.shape[1]
    n_k = K.shape[0]

    m = np.full(n_q, -np.inf, dtype=np.float64)
    l = np.zeros(n_q, dtype=np.float64)
    acc = np.zeros((n_q, d_v), dtype=np.float64)

    for start in range(0, n_k, block_size):
        end = min(start + block_size, n_k)
        K_blk = K[start:end]
        V_blk = V[start:end]

        scores = Q @ K_blk.T                       # (n_q, bs)
        block_max = np.max(scores, axis=1)          # (n_q,)
        new_m = np.maximum(m, block_max)

        # Rescale the OLD accumulators onto the NEW running max before
        # folding in this block's contribution. exp(-inf - finite) == 0,
        # so the very first block naturally zeroes out the (empty) history.
        alpha = np.exp(m - new_m)

        p = np.exp(scores - new_m[:, None])          # (n_q, bs)

        l = l * alpha + p.sum(axis=1)
        acc = acc * alpha[:, None] + p @ V_blk
        m = new_m

    return acc / l[:, None]
