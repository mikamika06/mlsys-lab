import math


def mha_gqa_mqa_reconstruct(Q: list[list[list[list[float]]]], K: list[list[list[list[float]]]], V: list[list[list[list[float]]]], group_sizes):
    """
    Reference: same input Q/K/V through several KV-grouping arities.

    For each group_size g in group_sizes, pool K and V within each group of
    g adjacent heads (mean), broadcast the pooled K/V back to n_heads heads,
    and run standard scaled dot-product attention with the original Q.
    g == 1 reproduces exact MHA; g == n_heads is MQA; anything in between is
    GQA(g).
    """
    batch = len(Q)
    seq_q = len(Q[0])
    n_heads = len(Q[0][0])
    d = len(Q[0][0][0])
    seq_k = len(K[0])
    sqrt_d = math.sqrt(d)

    results = []
    for g in group_sizes:
        n_kv = n_heads // g

        Kg = [[[[0.0 for _ in range(d)] for _ in range(n_kv)] for _ in range(seq_k)] for _ in range(batch)]
        Vg = [[[[0.0 for _ in range(d)] for _ in range(n_kv)] for _ in range(seq_k)] for _ in range(batch)]

        for b in range(batch):
            for sk in range(seq_k):
                for kv in range(n_kv):
                    head_start = kv * g
                    for di in range(d):
                        s_k = 0.0
                        s_v = 0.0
                        for gi in range(g):
                            s_k += K[b][sk][head_start + gi][di]
                            s_v += V[b][sk][head_start + gi][di]
                        Kg[b][sk][kv][di] = s_k / g
                        Vg[b][sk][kv][di] = s_v / g

        out = [[[[0.0 for _ in range(d)] for _ in range(n_heads)] for _ in range(seq_q)] for _ in range(batch)]
        weights_sk = [0.0] * seq_k

        for b in range(batch):
            for h in range(n_heads):
                kv = h // g
                for sq in range(seq_q):
                    max_score = 0.0
                    for sk in range(seq_k):
                        score = 0.0
                        for di in range(d):
                            score += Q[b][sq][h][di] * Kg[b][sk][kv][di]
                        score /= sqrt_d
                        weights_sk[sk] = score
                        if sk == 0 or score > max_score:
                            max_score = score

                    sum_exp = 0.0
                    for sk in range(seq_k):
                        e = math.exp(weights_sk[sk] - max_score)
                        weights_sk[sk] = e
                        sum_exp += e

                    for sk in range(seq_k):
                        weights_sk[sk] /= sum_exp

                    for di in range(d):
                        val = 0.0
                        for sk in range(seq_k):
                            val += weights_sk[sk] * Vg[b][sk][kv][di]
                        out[b][sq][h][di] = val

        size_ratio = n_kv / n_heads
        results.append((out, size_ratio))

    return results
