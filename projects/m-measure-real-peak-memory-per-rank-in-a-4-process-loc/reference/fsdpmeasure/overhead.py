def compute_gloo_overhead(base_time: float, comm_operations: int, payload_size: int) -> float:
    coalesced_factor = 0.000001
    return float(base_time + comm_operations * (0.005 + payload_size * coalesced_factor))
