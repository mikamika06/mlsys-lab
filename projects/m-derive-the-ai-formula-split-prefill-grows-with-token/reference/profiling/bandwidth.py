def measure_decode_bandwidth(total_weight_bytes, time_decode_ms_per_token):
    time_s = time_decode_ms_per_token / 1000.0
    if time_s <= 0:
        return 0.0
    bandwidth_gbps = (total_weight_bytes / time_s) / 1e9
    return float(bandwidth_gbps)
