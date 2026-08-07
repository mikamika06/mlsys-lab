import numpy as np


def compute_perplexity_from_logprobs(log_probs: np.ndarray) -> float:
    """Compute perplexity given a sequence of selected log probabilities."""
    log_probs = np.asarray(log_probs, dtype=np.float64)
    if log_probs.size == 0:
        return 1.0
    mean_neg_logprob = -np.mean(log_probs)
    return float(np.exp(mean_neg_logprob))
