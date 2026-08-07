def classify_slices(events):
    """Classify trace slices into category counts."""
    counts = {"compute": 0, "memory": 0, "communication": 0, "overhead": 0}
    for ev in events:
        name = ev.get("name", "")
        if "matmul" in name or "gemm" in name:
            counts["compute"] += 1
        elif "memcpy" in name or "mem" in name:
            counts["memory"] += 1
        elif "nccl" in name or "comm" in name:
            counts["communication"] += 1
        else:
            counts["overhead"] += 1
    return counts
