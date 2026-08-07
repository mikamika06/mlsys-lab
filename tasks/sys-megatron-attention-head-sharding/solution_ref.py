import math


def sharded_attention_heads(
    q: list[list[list[list[float]]]],
    k: list[list[list[list[float]]]],
    v: list[list[list[list[float]]]],
    wo: list[list[float]],
    num_ranks: int,
) -> list[list[list[float]]]:
    b = len(q)
    h = len(q[0])
    s = len(q[0][0])
    d = len(q[0][0][0])
    wo_dim = len(wo[0])

    out = [[[0.0 for _ in range(wo_dim)] for _ in range(s)] for _ in range(b)]
    heads_per_rank = h // num_ranks
    scale = math.sqrt(float(d))

    for rank in range(num_ranks):
        start = rank * heads_per_rank
        end = start + heads_per_rank
        partial = [[[0.0 for _ in range(wo_dim)] for _ in range(s)] for _ in range(b)]

        for head in range(start, end):
            scores = [[[0.0 for _ in range(s)] for _ in range(s)] for _ in range(b)]
            for bi in range(b):
                q_bi_h = q[bi][head]
                k_bi_h = k[bi][head]
                for i in range(s):
                    q_bi_h_i = q_bi_h[i]
                    for j in range(s):
                        k_bi_h_j = k_bi_h[j]
                        dot_val = 0.0
                        for l in range(d):
                            dot_val += q_bi_h_i[l] * k_bi_h_j[l]
                        scores[bi][i][j] = dot_val / scale

            max_scores = [[0.0 for _ in range(s)] for _ in range(b)]
            for bi in range(b):
                scores_bi = scores[bi]
                max_scores_bi = max_scores[bi]
                for i in range(s):
                    scores_bi_i = scores_bi[i]
                    m_val = scores_bi_i[0]
                    for j in range(1, s):
                        if scores_bi_i[j] > m_val:
                            m_val = scores_bi_i[j]
                    max_scores_bi[i] = m_val

            probs = [[[0.0 for _ in range(s)] for _ in range(s)] for _ in range(b)]
            sum_probs = [[0.0 for _ in range(s)] for _ in range(b)]
            for bi in range(b):
                scores_bi = scores[bi]
                max_scores_bi = max_scores[bi]
                probs_bi = probs[bi]
                sum_probs_bi = sum_probs[bi]
                for i in range(s):
                    scores_bi_i = scores_bi[i]
                    max_val = max_scores_bi[i]
                    probs_bi_i = probs_bi[i]
                    s_val = 0.0
                    for j in range(s):
                        val = math.exp(scores_bi_i[j] - max_val)
                        probs_bi_i[j] = val
                        s_val += val
                    sum_probs_bi[i] = s_val

            for bi in range(b):
                probs_bi = probs[bi]
                sum_probs_bi = sum_probs[bi]
                for i in range(s):
                    s_val = sum_probs_bi[i]
                    probs_bi_i = probs_bi[i]
                    for j in range(s):
                        probs_bi_i[j] /= s_val

            head_out = [[[0.0 for _ in range(d)] for _ in range(s)] for _ in range(b)]
            for bi in range(b):
                probs_bi = probs[bi]
                v_bi_h = v[bi][head]
                head_out_bi = head_out[bi]
                for i in range(s):
                    probs_bi_i = probs_bi[i]
                    head_out_bi_i = head_out_bi[i]
                    for l in range(d):
                        acc = 0.0
                        for j in range(s):
                            acc += probs_bi_i[j] * v_bi_h[j][l]
                        head_out_bi_i[l] = acc

            row_start = head * d
            for bi in range(b):
                head_out_bi = head_out[bi]
                partial_bi = partial[bi]
                for i in range(s):
                    head_out_bi_i = head_out_bi[i]
                    partial_bi_i = partial_bi[i]
                    for m_idx in range(wo_dim):
                        acc_wo = 0.0
                        for l in range(d):
                            acc_wo += head_out_bi_i[l] * wo[row_start + l][m_idx]
                        partial_bi_i[m_idx] += acc_wo

        for bi in range(b):
            out_bi = out[bi]
            partial_bi = partial[bi]
            for i in range(s):
                out_bi_i = out_bi[i]
                partial_bi_i = partial_bi[i]
                for m_idx in range(wo_dim):
                    out_bi_i[m_idx] += partial_bi_i[m_idx]

    return out
