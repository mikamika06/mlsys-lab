def reduce_time_ratio(num_elements: int, dtype_bytes_a: int, dtype_bytes_b: int) -> float:
    bw = 100e9
    latency = 1e-6
    time_a = latency + (num_elements * dtype_bytes_a) / bw
    time_b = latency + (num_elements * dtype_bytes_b) / bw
    return float(time_a / time_b)
