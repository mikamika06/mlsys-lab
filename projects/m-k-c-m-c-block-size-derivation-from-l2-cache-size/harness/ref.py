"""Reference oracle definitions for harness checking."""

import numpy as np

REGISTER_CONFIGS = [
    {"num_registers": 16, "vector_len_bytes": 32, "elem_size_bytes": 4},
    {"num_registers": 32, "vector_len_bytes": 64, "elem_size_bytes": 4},
    {"num_registers": 32, "vector_len_bytes": 64, "elem_size_bytes": 2},
    {"num_registers": 16, "vector_len_bytes": 16, "elem_size_bytes": 4},
]

CACHE_CONFIGS = [
    {"l2_size_bytes": 256 * 1024, "m_r": 4, "n_r": 16, "elem_size_bytes": 4, "alpha": 0.75},
    {"l2_size_bytes": 512 * 1024, "m_r": 6, "n_r": 32, "elem_size_bytes": 4, "alpha": 0.80},
    {"l2_size_bytes": 1024 * 1024, "m_r": 8, "n_r": 32, "elem_size_bytes": 2, "alpha": 0.70},
]


def ref_derive_register_tile(num_registers, vector_len_bytes, elem_size_bytes):
    vec_elems = vector_len_bytes // elem_size_bytes
    reserved_for_b = 1
    reserved_for_a = 1
    avail = num_registers - (reserved_for_a + reserved_for_b)
    if avail < 1:
        return (1, vec_elems)

    best_mr = 1
    best_nr_vec = 1
    best_product = 0

    max_mr = avail
    for mr in range(1, max_mr + 1):
        for nr_vec in range(1, avail // mr + 1):
            if mr * nr_vec <= avail:
                prod = mr * nr_vec
                if prod > best_product or (
                    prod == best_product and abs(mr - nr_vec) < abs(best_mr - best_nr_vec)
                ):
                    best_product = prod
                    best_mr = mr
                    best_nr_vec = nr_vec

    return (best_mr, best_nr_vec * vec_elems)


def ref_derive_l2_blocking(l2_size_bytes, m_r, n_r, elem_size_bytes, alpha=0.75):
    effective_l2 = int(l2_size_bytes * alpha)
    elements_budget = effective_l2 // elem_size_bytes

    k_c = 256
    while k_c * (n_r + m_r) > elements_budget and k_c > 8:
        k_c //= 2

    m_c_raw = (elements_budget - k_c * n_r) // k_c
    if m_c_raw < m_r:
        m_c = m_r
    else:
        m_c = (m_c_raw // m_r) * m_r

    return (m_c, k_c)


def ref_matmul_naive(a, b):
    return np.matmul(a, b)


def ref_matmul_cache_blocked(a, b, m_r, n_r, m_c, k_c):
    return np.matmul(a, b)
