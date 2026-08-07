COMM_LOG_FIXTURES = [
    {"log_entries": [{"op": "all_to_all", "volume": 100}], "P": 4, "S": 1024, "h": 512, "b": 1, "L": 12, "bytes_per_elem": 2},
    {"log_entries": [{"op": "all_to_all", "volume": 9437184}], "P": 4, "S": 1024, "h": 512, "b": 1, "L": 12, "bytes_per_elem": 2},
    {"log_entries": [{"op": "all_to_all", "volume": 0}], "P": 1, "S": 1024, "h": 512, "b": 1, "L": 12, "bytes_per_elem": 2},
]

MEM_BUDGET_FIXTURES = [
    {"mem_budget": 10 * 1024**3, "P": 4, "h": 512, "b": 1, "L": 12, "bytes_per_elem": 2},
    {"mem_budget": 40 * 1024**3, "P": 8, "h": 1024, "b": 2, "L": 24, "bytes_per_elem": 2},
    {"mem_budget": 1000, "P": 4, "h": 512, "b": 1, "L": 12, "bytes_per_elem": 2},
]

def verify_comm_log(log_entries, P, S, h, b, L, bytes_per_elem):
    expected = L * ((P - 1) / P) * (4 * S * b * h / P) * bytes_per_elem
    actual = sum(e["volume"] for e in log_entries if e.get("op") == "all_to_all")
    if expected == 0:
        return 0.0 if actual == 0 else float('inf')
    return abs(actual - expected) / expected

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
