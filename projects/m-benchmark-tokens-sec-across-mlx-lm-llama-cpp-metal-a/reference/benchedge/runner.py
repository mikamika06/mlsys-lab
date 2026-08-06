from typing import Any, Callable, Dict
from benchedge.metrics import BenchmarkResult, compute_benchmark_metrics


def run_backend_benchmark(
    backend: str,
    prompt_tokens: int,
    generated_tokens: int,
    trace_fn: Callable[[str, int, int], Dict[str, Any]],
) -> BenchmarkResult:
    trace = trace_fn(backend, prompt_tokens, generated_tokens)
    return compute_benchmark_metrics(
        backend=backend,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        t_start=trace["t_start"],
        t_first_token=trace["t_first_token"],
        t_end=trace["t_end"],
        rss_samples=trace["rss_samples"],
    )
