def simulate_tp_sweep(
    config: dict, tp_list: list[int], workload: dict, hardware: dict
) -> list[dict]:
    raise NotImplementedError


def find_optimal_tp(sweep_results: list[dict], metric: str = "throughput") -> int:
    raise NotImplementedError
