"""Naive and cache-blocked dense matrix multiplication."""

import numpy as np


def matmul_naive(a, b):
    m, k = a.shape
    k_b, n = b.shape
    assert k == k_b
    c = np.zeros((m, n), dtype=a.dtype)
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for p in range(k):
                acc += a[i, p] * b[p, j]
            c[i, j] = acc
    return c


def matmul_cache_blocked(a, b, m_r, n_r, m_c, k_c):
    m, k = a.shape
    k_b, n = b.shape
    assert k == k_b
    c = np.zeros((m, n), dtype=a.dtype)

    for i0 in range(0, m, m_c):
        i1 = min(i0 + m_c, m)
        for p0 in range(0, k, k_c):
            p1 = min(p0 + k_c, k)
            a_block = a[i0:i1, p0:p1]
            for j0 in range(0, n, n_r):
                j1 = min(j0 + n_r, n)
                b_block = b[p0:p1, j0:j1]

                for ii in range(0, i1 - i0, m_r):
                    ii_end = min(ii + m_r, i1 - i0)
                    a_tile = a_block[ii:ii_end, :]
                    c_sub = c[i0 + ii : i0 + ii_end, j0:j1]

                    c_sub += a_tile @ b_block

    return c
