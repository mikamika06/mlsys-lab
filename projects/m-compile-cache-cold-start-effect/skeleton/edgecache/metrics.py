import numpy as np


def compute_population_p95(fleet_data: list) -> float:
    """
    Compute population-weighted p95 inference latency across device groups.

    fleet_data is a list of dicts:
      [{"device_count": int, "latencies": list[float]}, ...]
    """
    raise NotImplementedError
