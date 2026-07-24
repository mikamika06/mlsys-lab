import numpy as np


def naive_merge(chunk_scores, chunk_values):
    """
    chunk_scores: list of C 1-D arrays; chunk_scores[i] holds the RAW
        (unstabilized) attention scores for chunk i, length n_i.
    chunk_values: list of C 2-D arrays; chunk_values[i] has shape (n_i, d) --
        the value vectors for that chunk's positions.

    Compute the softmax-weighted average of ALL values, treating every
    chunk's scores as if they were one flat vector:

        output = sum_i sum_j exp(L_i[j]) * V_i[j]  /  sum_i sum_j exp(L_i[j])

    WITHOUT any max-subtraction. This is the numerically NAIVE route: for
    large-magnitude scores, exp() overflows to +inf, and the ratio degrades
    to inf/inf = NaN. Returns a (d,) vector.
    """
    num = None
    den = 0.0
    for L, V in zip(chunk_scores, chunk_values):
        L = np.asarray(L, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        e = np.exp(L)
        contrib = (e[:, None] * V).sum(axis=0)
        num = contrib if num is None else num + contrib
        den += float(np.sum(e))
    return num / den


def lse_merge(chunk_scores, chunk_values):
    """
    Same target quantity as `naive_merge`, computed via the numerically
    STABLE log-sum-exp merge used by FlashAttention / ring-attention to
    combine partial outputs from different key/value blocks:

      1. For each chunk i, locally stabilize using its OWN max m_i:
           s_i = sum_j exp(L_i[j] - m_i)
           o_i = sum_j exp(L_i[j] - m_i) * V_i[j]
      2. Combine the C locally-stabilized chunks by rescaling every chunk
         to a single GLOBAL max g = max_i(m_i):
           alpha_i = exp(m_i - g)          # always <= 1, never overflows
           total_sumexp = sum_i alpha_i * s_i
           total_output = sum_i alpha_i * o_i
      3. output = total_output / total_sumexp

    Mathematically identical to naive_merge's target, but stable for any
    input magnitude. Returns a (d,) vector.
    """
    ms, ss, os_ = [], [], []
    for L, V in zip(chunk_scores, chunk_values):
        L = np.asarray(L, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        m = float(np.max(L))
        e = np.exp(L - m)
        ms.append(m)
        ss.append(float(np.sum(e)))
        os_.append((e[:, None] * V).sum(axis=0))

    ms = np.array(ms, dtype=np.float64)
    ss = np.array(ss, dtype=np.float64)
    os_ = np.array(os_, dtype=np.float64)

    gmax = float(np.max(ms))
    alpha = np.exp(ms - gmax)
    total_sumexp = float(np.sum(alpha * ss))
    total_output = (alpha[:, None] * os_).sum(axis=0)
    return total_output / total_sumexp
