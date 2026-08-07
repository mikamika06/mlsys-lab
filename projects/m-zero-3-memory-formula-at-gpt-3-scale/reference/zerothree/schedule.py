def simulate_all_gather_free_cycle(layer_sizes, dp_degree):
    timeline = []
    current_mem = 0
    peak_mem = 0
    for idx, size in enumerate(layer_sizes):
        gathered_size = size
        current_mem += gathered_size
        if current_mem > peak_mem:
            peak_mem = current_mem
        timeline.append({"layer": idx, "action": "all_gather", "memory": current_mem})
        current_mem -= (size - (size / dp_degree))
        timeline.append({"layer": idx, "action": "free", "memory": current_mem})
    return {"timeline": timeline, "peak_memory": float(peak_mem)}
