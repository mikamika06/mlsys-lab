import numpy as np

def streaming_softmax(scores: np.ndarray, acc=None):
    """
    Compute the softmax probabilities for a batch of scores while maintaining
    an online accumulator (maximum and sum of exponentials relative to that maximum).

    Parameters
    ----------
    scores : np.ndarray
        1-D array of float scores.
    acc : tuple or None
        Optional running state `(m, S)`. If None, start from scratch.

    Returns
    -------
    probs : np.ndarray
        Softmax probabilities for the input batch.
    new_acc : tuple
        Updated accumulator `(new_m, new_S)` that can be reused in a next call.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if acc is None:
        m = -np.inf
        S = 0.0
    else:
        m, S = acc

    for x in scores:
        if x > m:
            # rescale previous sum to the new maximum
            if m != -np.inf:
                S *= np.exp(m - x)
            m = x
        S += np.exp(x - m)

    probs = np.exp(scores - m) / S
    return probs, (m, S)
