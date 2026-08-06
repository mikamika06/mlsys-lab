import math
import numpy as np


def flash_attention_forward(Q, K, V, block_size=32):
    """Flash attention forward pass using online softmax, no NxN buffer."""
    Q = np.asarray(Q, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    V = np.asarray(V, dtype=np.float32)
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    out = np.zeros((N, d), dtype=np.float32)

    for i in range(N):
        q = Q[i]
        acc = np.zeros(d, dtype=np.float64)
        m = float('-inf')
        l = 0.0

        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            K_block = K[start:end]
            V_block = V[start:end]
            B = end - start

            scores = [0.0] * B
            for b in range(B):
                dot_val = 0.0
                for k in range(d):
                    dot_val += float(K_block[b, k]) * float(q[k])
                scores[b] = dot_val * scale

            max_score = float('-inf')
            for b in range(B):
                if scores[b] > max_score:
                    max_score = scores[b]

            m_new = m if m > max_score else max_score

            rescale = math.exp(m - m_new)
            for j in range(d):
                acc[j] *= rescale
            l *= rescale

            weights = [0.0] * B
            for b in range(B):
                weights[b] = math.exp(float(scores[b]) - m_new)

            for j in range(d):
                s_acc = 0.0
                for b in range(B):
                    s_acc += weights[b] * float(V_block[b, j])
                acc[j] += s_acc

            for b in range(B):
                l += weights[b]

            m = m_new

        for j in range(d):
            out[i, j] = float(acc[j] / l)

    return out
