def simulate_schedule(requests: list[dict], policy: str) -> tuple[float, int]:
    raise NotImplementedError


def simulate_eviction(requests: list[dict], capacity: int, policy: str) -> float:
    raise NotImplementedError


def simulate_tiering(requests: list[dict], gpu_c: int, host_c: int) -> tuple[float, float]:
    raise NotImplementedError
