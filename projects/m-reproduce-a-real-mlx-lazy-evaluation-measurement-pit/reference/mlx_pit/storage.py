def compute_size_ratio(adapter_bytes, base_bytes):
    if base_bytes == 0:
        return 0.0
    return float(adapter_bytes) / float(base_bytes)
