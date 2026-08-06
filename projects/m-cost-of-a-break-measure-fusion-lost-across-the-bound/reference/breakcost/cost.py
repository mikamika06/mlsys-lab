def compute_lost_cost(nodes, lost_fusions):
    total_bytes = 0
    for p in lost_fusions:
        idx = p[1]
        total_bytes += nodes[idx].get("bytes", 0)
    return total_bytes
