def estimate_optimization_speedup(
    aggregated: dict[str, dict],
    hw_spec: dict,
    target_kernels: list[str] = None,
    memory_reduction_factor: float = 0.0,
    target_efficiency: float = 1.0
) -> dict:
    raise NotImplementedError


def validate_prediction_against_profile(predicted_stats: dict, actual_records: list[dict]) -> dict:
    raise NotImplementedError


def generate_prioritized_report(aggregated: dict[str, dict], hw_spec: dict) -> list[dict]:
    raise NotImplementedError
