import numpy as np


def compute_acceptance_probs(target_probs, draft_probs):
    """Compute per-position acceptance probability alpha_i."""
    return np.sum(np.minimum(target_probs, draft_probs), axis=-1)


def expected_accepted_tokens(acceptance_probs):
    """Compute expected number of accepted tokens given acceptance probabilities."""
    probs = np.asarray(acceptance_probs, dtype=float)
    if len(probs) == 0:
        return 1.0
    cum_prod = np.cumprod(probs)
    return 1.0 + float(np.sum(cum_prod))
