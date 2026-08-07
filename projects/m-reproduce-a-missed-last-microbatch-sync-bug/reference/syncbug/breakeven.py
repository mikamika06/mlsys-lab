def compute_breakeven_k(element_size_bytes: int, upcast_overhead_per_element: float, bandwidth_bytes_per_sec: float) -> int:
    diff_time_per_byte = (4 - 2) / bandwidth_bytes_per_sec
    diff_time_per_element = element_size_bytes * diff_time_per_byte
    if diff_time_per_element <= 0:
        return 1
    k = upcast_overhead_per_element / diff_time_per_element
    return int(max(1, round(k)))
