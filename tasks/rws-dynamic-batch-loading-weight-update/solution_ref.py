import math
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
    n = len(prior)
    w_list = []
    total_sum = 0.0
    for i in range(n):
        val = prior[i] * math.exp(eta * excess[i])
        w_list.append(val)
        total_sum += val
    
    result = []
    for i in range(n):
        result.append(w_list[i] / total_sum)
        
    return np.array(result, dtype=np.float64)
