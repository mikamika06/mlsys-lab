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
    # Numerically stable softmax
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(logits - max_logits)
    probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    # Cross‑entropy per sample
    log_probs = np.log(probs + 1e-12)          # avoid log(0)
    ce_per_sample = -log_probs[np.arange(len(targets)), targets]

    # Mean cross‑entropy and perplexity
    mean_ce = np.mean(ce_per_sample)
    perplexity = float(np.exp(mean_ce))
    return perplexity
