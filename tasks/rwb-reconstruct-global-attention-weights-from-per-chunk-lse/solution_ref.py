import math
import numpy as np


def reconstruct_global_weights(
    chunk_scores: np.ndarray, chunk_lse: np.ndarray, chunk_partial_out: np.ndarray
) -> np.ndarray:
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

    Returns the flattened (C * chunk_size,) weight vector, in chunk order.
    """
    chunk_scores = np.asarray(chunk_scores, dtype=np.float64)
    chunk_lse = np.asarray(chunk_lse, dtype=np.float64)

    C = chunk_lse.shape[0]
    global_max = chunk_lse[0]
    for i in range(1, C):
        if chunk_lse[i] > global_max:
            global_max = chunk_lse[i]

    sum_exp = 0.0
    for i in range(C):
        sum_exp += math.exp(chunk_lse[i] - global_max)
    global_lse = global_max + math.log(sum_exp)

    C_scores, chunk_size = chunk_scores.shape
    weights = np.empty((C_scores, chunk_size), dtype=np.float64)
    for i in range(C_scores):
        for j in range(chunk_size):
            weights[i, j] = math.exp(chunk_scores[i, j] - global_lse)

    return weights.reshape(-1)
