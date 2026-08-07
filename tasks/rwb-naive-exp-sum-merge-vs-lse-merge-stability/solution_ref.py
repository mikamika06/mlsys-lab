import math


def naive_merge(
    chunk_scores: list[list[float]], chunk_values: list[list[list[float]]]
) -> list[float]:
    """
    chunk_scores: list of C 1-D lists; chunk_scores[i] holds the RAW
        (unstabilized) attention scores for chunk i, length n_i.
    chunk_values: list of C 2-D lists; chunk_values[i] has shape (n_i, d) --
        the value vectors for that chunk's positions.

    Compute the softmax-weighted average of ALL values, treating every
    chunk's scores as if they were one flat vector:

        output = sum_i sum_j exp(L_i[j]) * V_i[j]  /  sum_i sum_j exp(L_i[j])

    WITHOUT any max-subtraction. This is the numerically NAIVE route: for
    large-magnitude scores, exp() overflows to +inf, and the ratio degrades
    to inf/inf = NaN. Returns a list of floats of length d.
    """
    d = len(chunk_values[0][0])
    num = [0.0] * d
    den = 0.0
    for L, V in zip(chunk_scores, chunk_values):
        for score, v in zip(L, V):
            try:
                e = math.exp(score)
            except OverflowError:
                e = float("inf")
            den += e
            for k in range(d):
                num[k] += e * v[k]

    if math.isinf(den):
        return [float("nan")] * d

    return [x / den for x in num]


def lse_merge(
    chunk_scores: list[list[float]], chunk_values: list[list[list[float]]]
) -> list[float]:
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
    input magnitude. Returns a list of floats of length d.
    """
    d = len(chunk_values[0][0])
    ms = []
    ss = []
    os_ = []
    for L, V in zip(chunk_scores, chunk_values):
        m = max(L)
        s_i = 0.0
        o_i = [0.0] * d
        for score, v in zip(L, V):
            e = math.exp(score - m)
            s_i += e
            for k in range(d):
                o_i[k] += e * v[k]
        ms.append(m)
        ss.append(s_i)
        os_.append(o_i)

    gmax = max(ms)
    total_sumexp = 0.0
    total_output = [0.0] * d
    for i in range(len(ms)):
        alpha_i = math.exp(ms[i] - gmax)
        total_sumexp += alpha_i * ss[i]
        for k in range(d):
            total_output[k] += alpha_i * os_[i][k]

    return [x / total_sumexp for x in total_output]
