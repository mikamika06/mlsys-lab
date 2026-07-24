import numpy as np

def retained_variance_for_slice(eigenvalues, s):
    """Return (k, retained_ratio) for a given eigenvalue spectrum and target fraction.

    Parameters
    ----------
    eigenvalues : array-like of float
        Non-negative eigenvalues sorted in descending order.
    s : float
        Target fraction of variance to retain, in (0, 1].

    Returns
    -------
    k : int
        Smallest number of components such that the cumulative variance
        fraction is >= s.
    retained_ratio : float
        The actual cumulative variance fraction after k components.
    """
    raise NotImplementedError("your code here")
