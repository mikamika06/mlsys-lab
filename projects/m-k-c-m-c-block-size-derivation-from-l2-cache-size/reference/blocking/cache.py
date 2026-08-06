"""Cache blocking dimension derivation."""


def derive_l2_blocking(l2_size_bytes, m_r, n_r, elem_size_bytes, alpha=0.75):
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
