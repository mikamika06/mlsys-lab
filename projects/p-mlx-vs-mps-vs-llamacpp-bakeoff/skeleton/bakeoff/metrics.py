def measure_phase_latencies(raw_times: list[float]) -> dict:
    raise NotImplementedError


def aggregate_runs(run_results: list[dict]) -> dict:
    raise NotImplementedError


def evaluate_recommendation(metrics_summary: dict) -> str:
    raise NotImplementedError
