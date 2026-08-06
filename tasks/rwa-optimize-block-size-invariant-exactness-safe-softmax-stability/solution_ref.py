import math
import numpy as np


def tiled_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, block_size: int) -> np.ndarray:
    """
    Tiled (flash-style) attention forward using the online-softmax
    recurrence: sweep key/value blocks of at most `block_size` rows,
    maintaining a running row max `m` and running normalizer `l`, rescaling
    the accumulator whenever the running max increases. This is exact
    (identical to the dense computation, up to float rounding) regardless
    of `block_size`, including when `block_size` doesn't evenly divide the
    sequence length (ragged last block).
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    O = np.zeros((N, d), dtype=np.float64)
    m = np.full(N, -float("inf"))
    l = np.zeros(N, dtype=np.float64)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        Kb = K[start:end]
        Vb = V[start:end]
        block_len = end - start

        S = np.zeros((N, block_len), dtype=np.float64)
        for i in range(N):
            for j in range(block_len):
                dot = 0.0
                for k in range(d):
                    dot += Q[i, k] * Kb[j, k]
                S[i, j] = dot * scale

        block_max = np.full(N, -float("inf"))
        for i in range(N):
            row_max = -float("inf")
            for j in range(block_len):
                val = S[i, j]
                if val > row_max:
                    row_max = val
            block_max[i] = row_max

        m_new = np.zeros(N, dtype=np.float64)
        for i in range(N):
            if m[i] > block_max[i]:
                m_new[i] = m[i]
            else:
                m_new[i] = block_max[i]

        alpha = np.zeros(N, dtype=np.float64)
        for i in range(N):
            if m[i] == -float("inf"):
                alpha[i] = 0.0
            else:
                alpha[i] = math.exp(m[i] - m_new[i])

        P = np.zeros((N, block_len), dtype=np.float64)
        for i in range(N):
            for j in range(block_len):
                P[i, j] = math.exp(S[i, j] - m_new[i])

        p_sum = np.zeros(N, dtype=np.float64)
        for i in range(N):
            row_sum = 0.0
            for j in range(block_len):
                row_sum += P[i, j]
            p_sum[i] = row_sum

        for i in range(N):
            l[i] = l[i] * alpha[i] + p_sum[i]

        pv_dot = np.zeros((N, d), dtype=np.float64)
        for i in range(N):
            for k in range(d):
                acc = 0.0
                for j in range(block_len):
                    acc += P[i, j] * Vb[j, k]
                pv_dot[i, k] = acc

        for i in range(N):
            for k in range(d):
                O[i, k] = O[i, k] * alpha[i] + pv_dot[i, k]

        for i in range(N):
            m[i] = m_new[i]

    result = np.zeros((N, d), dtype=np.float64)
    for i in range(N):
        inv_l = 1.0 / l[i]
        for k in range(d):
            result[i, k] = O[i, k] * inv_l

    return result


def softmax_stability_probe(scores: np.ndarray):
    """
    Compute the numerically stable (max-subtracted) row-wise softmax of
    `scores`, and separately check whether the naive (no max-subtraction)
    softmax would overflow to a non-finite value on this same input.

    Returns (stable_out, unstable_overflowed).
    """
    scores = np.asarray(scores, dtype=np.float64)
    shape = scores.shape
    ndim = scores.ndim

    if ndim == 1:
        n = shape[0]
        row_max = -float("inf")
        for j in range(n):
            if scores[j] > row_max:
                row_max = scores[j]

        shifted = np.zeros(n, dtype=np.float64)
        for j in range(n):
            shifted[j] = scores[j] - row_max

        stable_w = np.zeros(n, dtype=np.float64)
        for j in range(n):
            stable_w[j] = math.exp(shifted[j])

        stable_sum = 0.0
        for j in range(n):
            stable_sum += stable_w[j]

        stable_out = np.zeros(n, dtype=np.float64)
        for j in range(n):
            stable_out[j] = stable_w[j] / stable_sum

        unstable_overflowed = False
        try:
            unstable_w = np.zeros(n, dtype=np.float64)
            for j in range(n):
                unstable_w[j] = math.exp(scores[j])

            unstable_sum = 0.0
            for j in range(n):
                unstable_sum += unstable_w[j]

            unstable_out = np.zeros(n, dtype=np.float64)
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
        rows = 1
        for i in range(ndim - 1):
            rows *= shape[i]
        cols = shape[-1]

        scores_2d = scores.reshape((rows, cols))
        stable_out_2d = np.zeros((rows, cols), dtype=np.float64)
        unstable_overflowed = False

        for i in range(rows):
            row_max = -float("inf")
            for j in range(cols):
                if scores_2d[i, j] > row_max:
                    row_max = scores_2d[i, j]

            shifted = np.zeros(cols, dtype=np.float64)
            for j in range(cols):
                shifted[j] = scores_2d[i, j] - row_max

            stable_w = np.zeros(cols, dtype=np.float64)
            for j in range(cols):
                stable_w[j] = math.exp(shifted[j])

            stable_sum = 0.0
            for j in range(cols):
                stable_sum += stable_w[j]

            for j in range(cols):
                stable_out_2d[i, j] = stable_w[j] / stable_sum

            try:
                unstable_w = np.zeros(cols, dtype=np.float64)
                for j in range(cols):
                    unstable_w[j] = math.exp(scores_2d[i, j])

                unstable_sum = 0.0
                for j in range(cols):
                    unstable_sum += unstable_w[j]

                unstable_out = np.zeros(cols, dtype=np.float64)
                for j in range(cols):
                    unstable_out[j] = unstable_w[j] / unstable_sum

                for j in range(cols):
                    if not math.isfinite(unstable_out[j]):
                        unstable_overflowed = True
            except (OverflowError, ValueError):
                unstable_overflowed = True

        stable_out = stable_out_2d.reshape(shape)
        return stable_out, unstable_overflowed
