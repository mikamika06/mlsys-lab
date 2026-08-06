def compute_dedup_savings(tensors):
    seen = {}
    unique_bytes = 0
    total_bytes = sum(t["size"] for t in tensors)
    for t in tensors:
        h = t["hash"]
        if h not in seen:
            seen[h] = True
            unique_bytes += t["size"]
    return {"total_bytes": total_bytes, "unique_bytes": unique_bytes, "savings": total_bytes - unique_bytes}
