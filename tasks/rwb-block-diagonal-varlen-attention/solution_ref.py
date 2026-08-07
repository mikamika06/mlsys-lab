import math


def varlen_block_diagonal_attention(q: list[list[float]], k: list[list[float]], v: list[list[float]],
                                     cu_seqlens: list[int]) -> list[list[float]]:
    """Packed varlen self-attention: several sequences packed row-wise
    into (N, d) tensors, boundaries given by `cu_seqlens` (the standard
    FlashAttention-varlen / xformers convention). Each token attends only
    within its own segment (full, non-causal attention inside a segment,
    none across segments).

    q, k, v    : (N, d) list of lists.
    cu_seqlens : list of ints, cu_seqlens[0] == 0,
                 cu_seqlens[-1] == N. Sequence i occupies rows
                 cu_seqlens[i] : cu_seqlens[i+1].

    Returns (N, d) list of lists.
    """
    N = len(q)
    d = len(q[0]) if N > 0 else 0
    out = [[0.0] * d for _ in range(N)]
    sqrt_d = math.sqrt(d) if d > 0 else 1.0

    n_seqs = len(cu_seqlens) - 1
    for seq_idx in range(n_seqs):
        start = int(cu_seqlens[seq_idx])
        end = int(cu_seqlens[seq_idx + 1])

        for i in range(start, end):
            scores = []
            for j in range(start, end):
                dot_val = 0.0
                for c in range(d):
                    dot_val += float(q[i][c]) * float(k[j][c])
                scores.append(dot_val / sqrt_d)

            max_score = scores[0] if scores else 0.0
            for score in scores:
                if score > max_score:
                    max_score = score

            exp_scores = []
            sum_exp = 0.0
            for score in scores:
                e = math.exp(score - max_score)
                exp_scores.append(e)
                sum_exp += e

            probs = [e / sum_exp for e in exp_scores]

            for c in range(d):
                v_val = 0.0
                for j_idx, j in enumerate(range(start, end)):
                    v_val += probs[j_idx] * float(v[j][c])
                out[i][c] = v_val

    return out
