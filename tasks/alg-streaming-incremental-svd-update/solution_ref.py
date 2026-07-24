import numpy as np


def incremental_svd_update(U, S, Vt, X_new, k):
    current = U @ np.diag(S) @ Vt
    full = np.vstack([current, X_new])
    U2, S2, Vt2 = np.linalg.svd(full, full_matrices=False)
    return (
        U2[:, :k],
        S2[:k],
        Vt2[:k, :],
    )
