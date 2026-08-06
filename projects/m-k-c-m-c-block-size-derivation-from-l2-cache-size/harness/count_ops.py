"""Helper module to evaluate memory access count efficiency."""


def count_naive_element_accesses(m, k, n):
    return 2 * m * k * n


def count_blocked_element_accesses(m, k, n, m_r, n_r, m_c, k_c):
    total_accesses = 0
    for i0 in range(0, m, m_c):
        i1 = min(i0 + m_c, m)
        mc_actual = i1 - i0
        for p0 in range(0, k, k_c):
            p1 = min(p0 + k_c, k)
            kc_actual = p1 - p0
            for j0 in range(0, n, n_r):
                j1 = min(j0 + n_r, n)
                nr_actual = j1 - j0

                total_accesses += kc_actual * nr_actual
                total_accesses += mc_actual * kc_actual

    return total_accesses
