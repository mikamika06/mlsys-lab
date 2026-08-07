def verify_comm_log(log_entries, P, S, h, b, L, bytes_per_elem):
    expected = L * ((P - 1) / P) * (4 * S * b * h / P) * bytes_per_elem
    actual = sum(e["volume"] for e in log_entries if e.get("op") == "all_to_all")
    if expected == 0:
        return 0.0 if actual == 0 else float('inf')
    return abs(actual - expected) / expected
