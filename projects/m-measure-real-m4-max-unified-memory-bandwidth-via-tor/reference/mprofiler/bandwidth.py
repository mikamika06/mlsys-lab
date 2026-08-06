def compute_bandwidth_percentage(bytes_transferred, duration_s, peak_gbps=546.0):
    if duration_s <= 0:
        return 0.0
    achieved_gbps = (bytes_transferred / 1e9) / duration_s
    return float((achieved_gbps / peak_gbps) * 100.0)
