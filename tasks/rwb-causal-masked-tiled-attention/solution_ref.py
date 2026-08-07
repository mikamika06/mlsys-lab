import math

def tiled_causal_attention(Q, K, V, block_size):
    B = len(Q)
    H = len(Q[0])
    N = len(Q[0][0])
    D = len(Q[0][0][0])

    O = [[[[0.0 for _ in range(D)] for _ in range(N)] for _ in range(H)] for _ in range(B)]
    scale = 1.0 / math.sqrt(D)

    for b in range(B):
        for h in range(H):
            for q_start in range(0, N, block_size):
                q_end = min(q_start + block_size, N)

                # running max and denominator for the current block
                m = [-float('inf')] * (q_end - q_start)
                l = [0.0] * (q_end - q_start)
                out = [[0.0] * D for _ in range(q_end - q_start)]

                for k_start in range(0, N, block_size):
                    k_end = min(k_start + block_size, N)

                    if k_start > q_end - 1:
                        continue

                    for i in range(q_start, q_end):
                        i_idx = i - q_start
                        for j in range(k_start, k_end):
                            if j > i:
                                continue

                            s_ij = sum(Q[b][h][i][d] * K[b][h][j][d] for d in range(D)) * scale

                            m_prev = m[i_idx]
                            m_new = max(m_prev, s_ij)

                            exp_prev = math.exp(m_prev - m_new) if m_prev != -float('inf') else 0.0
                            exp_new = math.exp(s_ij - m_new)

                            l[i_idx] = l[i_idx] * exp_prev + exp_new

                            for d in range(D):
                                out[i_idx][d] = out[i_idx][d] * exp_prev + exp_new * V[b][h][j][d]

                            m[i_idx] = m_new

                for i in range(q_start, q_end):
                    i_idx = i - q_start
                    for d in range(D):
                        if l[i_idx] > 0:
                            O[b][h][i][d] = out[i_idx][d] / l[i_idx]

    return O
