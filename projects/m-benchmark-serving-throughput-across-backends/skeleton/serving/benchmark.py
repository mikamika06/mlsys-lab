def run_benchmark_pass(backend: str, batch_size: int, prompt_len: int, gen_len: int) -> dict[str, float]:
    """Simulate serving execution for a backend pass and return performance metrics."""
    raise NotImplementedError


def generate_tradeoff_report(results: list[dict[str, float]]) -> dict[str, list[dict[str, float]]]:
    """Structure raw benchmark pass metrics into TTFT vs ITL tradeoff report."""
    raise NotImplementedError
