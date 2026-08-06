import math
import numpy as np


def sharded_attention_heads(q, k, v, wo, num_ranks):
    b, h, s, d = q.shape
    out = np.zeros((b, s, wo.shape[1]), dtype=np.float64)
    heads_per_rank = h // num_ranks
    scale = math.sqrt(float(d))
    wo_dim = wo.shape[1]

    for rank in range(num_ranks):
        start = rank * heads_per_rank
        end = start + heads_per_rank
        partial = np.zeros_like(out)

        for head in range(start, end):
            q_h = q[:, head]
            k_h = k[:, head]
            v_h = v[:, head]

            scores = np.zeros((b, s, s), dtype=np.float64)
            for bi in range(b):
                for i in range(s):
                    for j in range(s):
                        dot_val = 0.0
                        for l in range(d):
                            dot_val += q_h[bi, i, l] * k_h[bi, j, l]
                        scores[bi, i, j] = dot_val / scale

            max_scores = np.zeros((b, s, 1), dtype=np.float64)
            for bi in range(b):
                for i in range(s):
                    m_val = scores[bi, i, 0]
                    for j in range(1, s):
                        if scores[bi, i, j] > m_val:
                            m_val = scores[bi, i, j]
                    max_scores[bi, i, 0] = m_val

            probs = np.zeros((b, s, s), dtype=np.float64)
            sum_probs = np.zeros((b, s, 1), dtype=np.float64)
            for bi in range(b):
                for i in range(s):
                    s_val = 0.0
                    for j in range(s):
                        val = math.exp(scores[bi, i, j] - max_scores[bi, i, 0])
                        probs[bi, i, j] = val
                        s_val += val
                    sum_probs[bi, i, 0] = s_val

            for bi in range(b):
                for i in range(s):
                    for j in range(s):
                        probs[bi, i, j] /= sum_probs[bi, i, 0]

            head_out = np.zeros((b, s, d), dtype=np.float64)
            for bi in range(b):
                for i in range(s):
                    for l in range(d):
                        acc = 0.0
                        for j in range(s):
                            acc += probs[bi, i, j] * v_h[bi, j, l]
                        head_out[bi, i, l] = acc

            row_start = head * d
            for bi in range(b):
                for i in range(s):
                    for m_idx in range(wo_dim):
                        acc_wo = 0.0
                        for l in range(d):
                            acc_wo += head_out[bi, i, l] * wo[row_start + l, m_idx]
                        partial[bi, i, m_idx] += acc_wo

        for bi in range(b):
            for i in range(s):
                for m_idx in range(wo_dim):
                    out[bi, i, m_idx] += partial[bi, i, m_idx]

    return out
