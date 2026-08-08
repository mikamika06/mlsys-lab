def compute_hardware_penalty(
    engine_sm: tuple, target_sm: tuple, base_latency_ms: float
) -> float:
    e_major, e_minor = engine_sm
    t_major, t_minor = target_sm
    if (e_major, e_minor) == (t_major, t_minor):
        return base_latency_ms
    if e_major != t_major:
        if t_major < e_major:
            return float("inf")
        major_diff = t_major - e_major
        return base_latency_ms * (1.5**major_diff) + 2.0
    minor_diff = abs(t_minor - e_minor)
    return base_latency_ms * (1.1**minor_diff)
