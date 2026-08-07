def evaluate_mse(w, w_pruned, bias, x):
    """
    Evaluates the Mean Squared Error between the original dense output
    and the pruned output (with bias correction).
    Returns a float.
    """
    raise NotImplementedError


def compare_methods(w, x, sparsity=0.5):
    """
    Compares Magnitude pruning (without bias correction) and
    Wanda pruning (with bias correction) at the given sparsity.
    Returns: (mse_magnitude, mse_wanda)
    """
    raise NotImplementedError
