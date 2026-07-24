import numpy as np


def _loss(Q):
    n = Q.shape[1]
    return float(np.max(np.abs(Q.T @ Q - np.eye(n, dtype=np.float64))))


def _cgs(A):
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float64)
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            v -= np.dot(Q[:, i], A[:, j]) * Q[:, i]
        Q[:, j] = v / np.linalg.norm(v)
    return Q


def _mgs(A):
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    V = A.copy()
    Q = np.zeros((m, n), dtype=np.float64)
    for i in range(n):
        q = V[:, i] / np.linalg.norm(V[:, i])
        Q[:, i] = q
        for j in range(i + 1, n):
            V[:, j] -= np.dot(q, V[:, j]) * q
    return Q


def gram_schmidt_orthogonality(A):
    A = np.asarray(A, dtype=np.float64)
    return _loss(_cgs(A)), _loss(_mgs(A))
