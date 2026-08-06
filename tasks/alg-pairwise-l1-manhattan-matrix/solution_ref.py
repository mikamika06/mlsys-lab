import math


def pairwise_l1_matrix(X: list[list[float]], Y: list[list[float]] | None = None) -> list[list[float]]:
    """
    Compute the pairwise Manhattan (L1) distance matrix between rows of X and Y.
    If Y is None, compute distances within X.

    Parameters
    ----------
    X : list[list[float]]
        2-D array of shape (n, d).
    Y : list[list[float]] | None, optional
        2-D array of shape (m, d). Defaults to None.

    Returns
    -------
    D : list[list[float]]
        2-D array of shape (n, m) containing L1 distances.
    """
    if Y is None:
        Y = X
    n = len(X)
    d = len(X[0]) if n > 0 else 0
    m = len(Y)
    D = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            dist = 0.0
            for k in range(d):
                diff = X[i][k] - Y[j][k]
                dist += math.fabs(diff)
            D[i][j] = dist
    return D
