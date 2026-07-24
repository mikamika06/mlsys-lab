import numpy as np


def flash_attention_forward(Q, K, V, block_size=32):
    """Flash attention forward pass using online softmax, no NxN buffer."""
    Q = np.asarray(Q, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    V = np.asarray(V, dtype=np.float32)
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    out = np.zeros((N, d), dtype=np.float32)

    for i in range(N):
        q = Q[i]  # (d,)
        acc = np.zeros(d, dtype=np.float64)
        m = -np.inf
        l = 0.0

        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            K_block = K[start:end]  # (B, d)
            V_block = V[start:end]  # (B, d)

            scores = (K_block @ q) * scale  # (B,)
            m_new = max(m, float(scores.max()))

            # Rescale accumulator
            rescale = np.exp(m - m_new)
            acc = acc * rescale
            l = l * rescale

            # Accumulate
            weights = np.exp(scores.astype(np.float64) - m_new)  # (B,)
            acc += weights @ V_block.astype(np.float64)
            l += weights.sum()

            m = m_new

        out[i] = (acc / l).astype(np.float32)

    return out
