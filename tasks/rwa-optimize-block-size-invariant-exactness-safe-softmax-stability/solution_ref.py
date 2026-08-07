from __future__ import annotations

import math


def tiled_attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]], block_size: int) -> list[list[float]]:
    """
    Tiled (flash-style) attention forward using the online-softmax
    recurrence: sweep key/value blocks of at most `block_size` rows,
    maintaining a running row max `m` and running normalizer `l`, rescaling
    the accumulator whenever the running max increases. This is exact
    (identical to the dense computation, up to float rounding) regardless
    of `block_size`, including when `block_size` doesn't evenly divide the
    sequence length (ragged last block).
    """
    N = len(Q)
    d = len(Q[0])
    scale = 1.0 / math.sqrt(d)

    O = [[0.0] * d for _ in range(N)]
    m = [-float("inf")] * N
    l = [0.0] * N

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        Kb = K[start:end]
        Vb = V[start:end]
        block_len = end - start

        S = [[0.0] * block_len for _ in range(N)]
        for i in range(N):
            for j in range(block_len):
                dot = 0.0
                for k in range(d):
                    dot += Q[i][k] * Kb[j][k]
                S[i][j] = dot * scale

        block_max = [-float("inf")] * N
        for i in range(N):
            row_max = -float("inf")
            for j in range(block_len):
                val = S[i][j]
                if val > row_max:
                    row_max = val
            block_max[i] = row_max

        m_new = [0.0] * N
        for i in range(N):
            if m[i] > block_max[i]:
                m_new[i] = m[i]
            else:
                m_new[i] = block_max[i]

        alpha = [0.0] * N
        for i in range(N):
            if m[i] == -float("inf"):
                alpha[i] = 0.0
            else:
                alpha[i] = math.exp(m[i] - m_new[i])

        P = [[0.0] * block_len for _ in range(N)]
        for i in range(N):
            for j in range(block_len):
                P[i][j] = math.exp(S[i][j] - m_new[i])

        p_sum = [0.0] * N
        for i in range(N):
            row_sum = 0.0
            for j in range(block_len):
                row_sum += P[i][j]
            p_sum[i] = row_sum

        for i in range(N):
            l[i] = l[i] * alpha[i] + p_sum[i]

        pv_dot = [[0.0] * d for _ in range(N)]
        for i in range(N):
            for k in range(d):
                acc = 0.0
                for j in range(block_len):
                    acc += P[i][j] * Vb[j][k]
                pv_dot[i][k] = acc

        for i in range(N):
            for k in range(d):
                O[i][k] = O[i][k] * alpha[i] + pv_dot[i][k]

        for i in range(N):
            m[i] = m_new[i]

    result = [[0.0] * d for _ in range(N)]
    for i in range(N):
        inv_l = 1.0 / l[i]
        for k in range(d):
            result[i][k] = O[i][k] * inv_l

    return result


def softmax_stability_probe(scores: list[float] | list[list[float]]):
    """
    Compute the numerically stable (max-subtracted) row-wise softmax of
    `scores`, and separately check whether the naive (no max-subtraction)
    softmax would overflow to a non-finite value on this same input.

    Returns (stable_out, unstable_overflowed).
    """
    if isinstance(scores[0], (int, float)):
        n = len(scores)
        row_max = -float("inf")
        for j in range(n):
            if scores[j] > row_max:
                row_max = scores[j]

        shifted = [0.0] * n
        for j in range(n):
            shifted[j] = scores[j] - row_max

        stable_w = [0.0] * n
        for j in range(n):
            stable_w[j] = math.exp(shifted[j])

        stable_sum = 0.0
        for j in range(n):
            stable_sum += stable_w[j]

        stable_out = [0.0] * n
        for j in range(n):
            stable_out[j] = stable_w[j] / stable_sum

        unstable_overflowed = False
        try:
            unstable_w = [0.0] * n
            for j in range(n):
                unstable_w[j] = math.exp(scores[j])

            unstable_sum = 0.0
            for j in range(n):
                unstable_sum += unstable_w[j]

            unstable_out = [0.0] * n
            for j in range(n):
                unstable_out[j] = unstable_w[j] / unstable_sum

            for j in range(n):
                if not math.isfinite(unstable_out[j]):
                    unstable_overflowed = True
                    break
        except (OverflowError, ValueError):
            unstable_overflowed = True

        return stable_out, unstable_overflowed

    else:
        rows = len(scores)
        cols = len(scores[0])
        stable_out_2d = [[0.0] * cols for _ in range(rows)]
        unstable_overflowed = False

        for i in range(rows):
            row_max = -float("inf")
            for j in range(cols):
                if scores[i][j] > row_max:
                    row_max = scores[i][j]

            shifted = [0.0] * cols
            for j in range(cols):
                shifted[j] = scores[i][j] - row_max

            stable_w = [0.0] * cols
            for j in range(cols):
                stable_w[j] = math.exp(shifted[j])

            stable_sum = 0.0
            for j in range(cols):
                stable_sum += stable_w[j]

            for j in range(cols):
                stable_out_2d[i][j] = stable_w[j] / stable_sum

            try:
                unstable_w = [0.0] * cols
                for j in range(cols):
                    unstable_w[j] = math.exp(scores[i][j])

                unstable_sum = 0.0
                for j in range(cols):
                    unstable_sum += unstable_w[j]

                unstable_out = [0.0] * cols
                for j in range(cols):
                    unstable_out[j] = unstable_w[j] / unstable_sum

                for j in range(cols):
                    if not math.isfinite(unstable_out[j]):
                        unstable_overflowed = True
            except (OverflowError, ValueError):
                unstable_overflowed = True

        return stable_out_2d, unstable_overflowed
