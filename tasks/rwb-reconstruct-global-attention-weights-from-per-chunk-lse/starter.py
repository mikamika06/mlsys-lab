import math

def reconstruct_global_weights(chunk_scores: list[list[float]], chunk_lse: list[float], chunk_partial_out: list[list[float]]) -> list[float]:
    """Reconstruct the global softmax attention weight of every KV token
    from per-chunk statistics (as a chunked/ring-attention worker would
    have on hand for one query row).

    chunk_scores: (C, chunk_size) array -- raw score for every token,
        grouped by which chunk it belongs to.
    chunk_lse: (C,) array -- each chunk's own log-sum-exp of its scores,
        i.e. chunk_lse[c] == logsumexp(chunk_scores[c]).
    chunk_partial_out: (C, d) array -- each chunk's local unnormalized
        partial output; provided for context (this is what a real
        chunked-attention worker also carries), not required to compute
        the weights.

    The token at global position (c, i) has global softmax weight
        w = exp(chunk_scores[c, i] - global_lse)
    where global_lse is the TRUE log-sum-exp over every token across every
    chunk. Because sum_all exp(score) == sum_c exp(chunk_lse[c]) exactly,
    global_lse can be recovered as logsumexp(chunk_lse) -- a log-sum-exp
    of only C numbers, without needing to re-scan every individual score.

    Returns the flattened (C * chunk_size,) weight vector, in the same
    (chunk, position-within-chunk) order as chunk_scores.reshape(-1).
    """
    raise NotImplementedError('your code here')
