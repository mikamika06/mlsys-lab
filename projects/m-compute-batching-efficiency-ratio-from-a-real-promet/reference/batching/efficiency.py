def compute_batching_efficiency(dump_str, max_batch_size=8.0):
    """Compute batching efficiency from Prometheus dump."""
    vals = []
    for line in dump_str.splitlines():
        if "nv_inference_request_batch_size" in line:
            parts = line.strip().split()
            if len(parts) == 2:
                vals.append(float(parts[1]))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals) / max_batch_size)
