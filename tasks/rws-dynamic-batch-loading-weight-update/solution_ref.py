import numpy as np

def update_weights(prior, excess, eta):
    """
    Update domain weights using exponentiated gradient (softmax) rule.

    Parameters
    ----------
    prior : array-like of shape (n,)
        Current domain weights; must sum to 1.
    excess : array-like of shape (n,)
        Excess loss for each domain. Positive values indicate higher loss.
    eta : float
        Learning rate controlling the magnitude of the update.

    Returns
    -------
    np.ndarray
        Updated weights, shape (n,), sum to 1, dtype float64.
    """
    prior = np.asarray(prior, dtype=np.float64)
    excess = np.asarray(excess, dtype=np.float64)
    w = prior * np.exp(eta * excess)
    return w / w.sum()
