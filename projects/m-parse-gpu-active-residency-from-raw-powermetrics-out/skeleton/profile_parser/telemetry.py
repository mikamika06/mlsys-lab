def correlate_residency_with_throughput(
    gpu_residencies: list[float], tokens_per_sec: list[float]
) -> dict:
    raise NotImplementedError


def estimate_ane_utilization(
    ane_powers_mw: list[float], max_ane_power_mw: float = 8000.0
) -> dict:
    raise NotImplementedError
