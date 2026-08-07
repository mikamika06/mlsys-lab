import math


def block_diagonal_attention(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
    seq_lens: list[int],
) -> list[list[float]]:
    """Attention over several variable-length sequences PACKED into one
    (N, d) tensor along the row axis (xformers' BlockDiagonalMask). Each
    sequence attends only to its own rows -- full (non-causal) attention
    within a sequence, zero cross-sequence attention.

    Q, K, V   : (N, d), N == sum(seq_lens).
    seq_lens  : list of positive ints, the length of each packed sequence,
                in order.

    Returns (N, d).
    """
    N = len(Q)
    d = len(Q[0]) if N > 0 else 0
    sqrt_d = math.sqrt(d)
    outs = []
    pos = 0
    for L in seq_lens:
        Qs = Q[pos : pos + L]
        Ks = K[pos : pos + L]
        Vs = V[pos : pos + L]

        scores = []
        for i in range(L):
            row = []
            for j in range(L):
                dot = 0.0
                for k in range(d):
                    dot += Qs[i][k] * Ks[j][k]
                row.append(dot / sqrt_d)
            scores.append(row)

        probs = []
        for i in range(L):
            max_val = max(scores[i])
            row_probs = []
            row_sum = 0.0
            for j in range(L):
                val = math.exp(scores[i][j] - max_val)
                row_probs.append(val)
                row_sum += val
            normalized_row = []
            for val in row_probs:
                normalized_row.append(val / row_sum)
            probs.append(normalized_row)

        block_out = []
        for i in range(L):
            out_row = []
            for c in range(d):
                acc = 0.0
                for j in range(L):
                    acc += probs[i][j] * Vs[j][c]
                out_row.append(acc)
            block_out.append(out_row)

        outs.extend(block_out)
        pos += L

    return outs
