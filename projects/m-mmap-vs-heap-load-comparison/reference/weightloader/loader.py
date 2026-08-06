def simulate_load(tensors, mode):
    total_bytes = sum(t["size"] for t in tensors)
    page_size = 4096
    if mode == "heap":
        peak_rss = total_bytes + sum((page_size - (t["size"] % page_size)) % page_size for t in tensors)
        shared_rss = 0
        page_faults = sum((t["size"] + page_size - 1) // page_size for t in tensors)
    elif mode == "mmap":
        peak_rss = page_size * len(tensors)
        shared_rss = total_bytes
        page_faults = len(tensors)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return {"peak_rss": peak_rss, "shared_rss": shared_rss, "page_faults": page_faults}
