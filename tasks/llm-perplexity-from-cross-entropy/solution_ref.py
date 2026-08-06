import math
import numpy as np

def perplexity_from_cross_entropy(logits: np.ndarray,
                                   targets: np.ndarray) -> float:
    """
    Compute the perplexity of a language model given raw logits and target indices.

    Parameters
    ----------
    logits : np.ndarray, shape (N, V)
        Raw scores for each token in the vocabulary.
    targets : np.ndarray, shape (N,)
        Integer indices of the correct token for each sample.

    Returns
    -------
    float
        The perplexity value as a scalar Python float.
    """
    N, V = logits.shape
    total_ce = 0.0

    for i in range(N):
        row = logits[i]
        target_idx = targets[i]

        max_val = row[0]
        for j in range(1, V):
            if row[j] > max_val:
                max_val = row[j]

        sum_exp = 0.0
        target_exp = 0.0
        for j in range(V):
            val = math.exp(row[j] - max_val)
            sum_exp += val
            if j == target_idx:
                target_exp = val

        prob = target_exp / sum_exp
        log_prob = math.log(prob + 1e-12)
        total_ce += -log_prob

    mean_ce = total_ce / N
    return float(math.exp(mean_ce))
