import math


def reconstruct_global_weights(
    chunk_scores: list[list[float]], chunk_lse: list[float], chunk_partial_out: list[list[float]]
) -> list[float]:
    """
    Reconstruct the global softmax attention weight for every KV token,
    given per-chunk statistics as a chunked/ring-attention worker would
    produce them for one query row:

      chunk_scores: (C, chunk_size) -- each token's raw score.
      chunk_lse: (C,) -- each chunk's own log-sum-exp of its scores
          (a single scalar reduction, e.g. exchanged between workers).
      chunk_partial_out: (C, d) -- each chunk's local unnormalized
          partial output (not needed to recover the weights themselves).

    The true global normalizer is the log-sum-exp of the per-chunk
    log-sum-exps (since sum_all exp(score) = sum_c exp(LSE_c) exactly):

        global_lse = logsumexp(chunk_lse)
        w_j = exp(score_j - global_lse)

    Returns the flattened (C * chunk_size,) weight list, in chunk order.
    """
    C = len(chunk_lse)
    if C == 0:
        return []

    global_max = chunk_lse[0]
    for i in range(1, C):
        if chunk_lse[i] > global_max:
            global_max = chunk_lse[i]

    sum_exp = 0.0
    for i in range(C):
        sum_exp += math.exp(chunk_lse[i] - global_max)
    global_lse = global_max + math.log(sum_exp)

    weights = []
    for chunk in chunk_scores:
        for score in chunk:
            weights.append(math.exp(score - global_lse))

    return weights
