def pick_rank_and_report(A):
    import numpy as np
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    energy = S**2
    cum_energy = np.cumsum(energy)
    total = cum_energy[-1]
    ratio = cum_energy / total
    k = int(np.searchsorted(ratio, 0.9) + 1)
    size_ratio = (A.shape[0]*k + k + k*A.shape[1])/(A.shape[0]*A.shape[1])
    return k, float(size_ratio)
