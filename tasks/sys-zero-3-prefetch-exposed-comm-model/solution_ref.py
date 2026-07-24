def exposed_comm_bytes(layer_bytes, compute_times, bandwidth, prefetch_depth):
    exposed = 0.0
    for i, size in enumerate(layer_bytes):
        start = max(0, i - prefetch_depth)
        overlap = sum(float(x) for x in compute_times[start:i])
        comm_time = float(size) / float(bandwidth)
        exposed += max(0.0, comm_time - overlap) * float(bandwidth)
    return float(exposed)
