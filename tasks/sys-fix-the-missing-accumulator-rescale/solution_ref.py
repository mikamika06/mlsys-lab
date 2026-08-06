import numpy as np
import math

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
        m = -float('inf')
        S = 0.0
    else:
        m, S = acc

    for x in scores:
        if x > m:
            if m != -float('inf'):
                S *= math.exp(m - x)
            m = x
        S += math.exp(x - m)

    probs = np.empty_like(scores, dtype=np.float64)
    for i in range(len(scores)):
        probs[i] = math.exp(scores[i] - m) / S

    return probs, (m, S)
