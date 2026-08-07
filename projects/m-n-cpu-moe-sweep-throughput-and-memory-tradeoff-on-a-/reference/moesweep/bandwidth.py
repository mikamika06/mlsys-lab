def derive_bandwidth(transferred_bytes, latencies, target_throughput):
    if not transferred_bytes or not latencies or len(transferred_bytes) != len(latencies):
        return 0.0
    bandwidths = [b / l if l > 0 else 0.0 for b, l in zip(transferred_bytes, latencies)]
    avg_bw = sum(bandwidths) / len(bandwidths)
    scaling_factor = target_throughput / max(1.0, sum(latencies))
    return avg_bw * max(1.0, scaling_factor)
