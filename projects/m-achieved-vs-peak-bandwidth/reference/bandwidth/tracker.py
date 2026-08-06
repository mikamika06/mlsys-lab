import math


def compute_bytes_transferred(config: dict) -> dict:
    """Calculate naive and tiled HBM bytes transferred."""
    b = config["batch_size"]
    h = config["num_heads"]
    n = config["seq_len"]
    d = config["head_dim"]
    p = config["element_bytes"]
    br = config["block_r"]

    tr = math.ceil(n / br)

    naive_bytes = float(b * h * p * (4 * n * d + 2 * n * n))
    tiled_bytes = float(b * h * n * d * p * (2 + 2 * tr))

    return {"naive_bytes": naive_bytes, "tiled_bytes": tiled_bytes}


def compute_achieved_bandwidth(bytes_transferred: float, execution_time_sec: float) -> float:
    """Compute achieved memory bandwidth in GB/s."""
    return float((bytes_transferred / 1e9) / execution_time_sec)


def compute_bandwidth_efficiency(achieved_gbps: float, peak_gbps: float) -> float:
    """Compute bandwidth utilization ratio relative to theoretical peak."""
    return float(achieved_gbps / peak_gbps)
