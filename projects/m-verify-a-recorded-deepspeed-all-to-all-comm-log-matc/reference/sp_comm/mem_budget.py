def max_seq_len(mem_budget, P, h, b, L, bytes_per_elem):
    m_static = 16 * L * 12 * (h ** 2)
    avail = mem_budget - m_static
    if avail < 0:
        return {"dense": 0, "ulysses": 0, "ring": 0}

    c_dense = L * 34 * b * h * bytes_per_elem
    s_dense = int(avail / c_dense) if c_dense > 0 else 0

    c_ulysses = (L * 34 * b * h * bytes_per_elem) / P + (3 * b * h * bytes_per_elem) / P
    s_ulysses = int(avail / c_ulysses) if c_ulysses > 0 else 0

    c_ring = (L * 34 * b * h * bytes_per_elem) / P + (2 * b * h * bytes_per_elem) / P
    s_ring = int(avail / c_ring) if c_ring > 0 else 0

    return {"dense": s_dense, "ulysses": s_ulysses, "ring": s_ring}
