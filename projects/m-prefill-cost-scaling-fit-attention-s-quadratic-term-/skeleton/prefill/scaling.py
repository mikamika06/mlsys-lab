import numpy as np

def fit_scaling(lengths, times):
    """
    Given a list of context lengths and corresponding prefill times,
    fit a quadratic curve time = a + b*L + c*L^2.

    Returns:
        dict: {"linear": b, "quadratic": c}
    """
    raise NotImplementedError
