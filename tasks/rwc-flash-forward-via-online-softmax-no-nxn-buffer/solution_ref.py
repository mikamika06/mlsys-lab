import math


def flash_attention_forward(Q: list[list[float]], K: list[list[float]], V: list[list[float]], block_size: int = 32) -> list[list[float]]:
    """Flash attention forward pass using online softmax, no NxN buffer."""
    N = len(Q)
    d = len(Q[0]) if N > 0 else 0
    scale = 1.0 / math.sqrt(d)
    out = [[0.0] * d for _ in range(N)]

    for i in range(N):
        q = Q[i]
        acc = [0.0] * d
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
                    dot_val += float(K_block[b][k]) * float(q[k])
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
                    s_acc += weights[b] * float(V_block[b][j])
                acc[j] += s_acc

            for b in range(B):
                l += weights[b]

            m = m_new

        for j in range(d):
            out[i][j] = float(acc[j] / l)

    return out
