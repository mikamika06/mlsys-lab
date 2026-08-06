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
    ev = np.asarray(eigenvalues, dtype=np.float64)
    ev = np.asarray(sorted(ev), dtype=np.float64)[::-1]
    
    total = 0.0
    for val in ev:
        total += float(val)
        
    if total <= 0.0:
        return (0, 0.0)
        
    cumsum = []
    current = 0.0
    for val in ev:
        current += float(val)
        cumsum.append(current)
        
    fractions = [c / total for c in cumsum]
    
    idx = len(fractions)
    for i, f in enumerate(fractions):
        if f >= s:
            idx = i
            break
            
    k = min(idx + 1, len(ev))
    retained = float(cumsum[k - 1] / total)
    return (k, retained)
