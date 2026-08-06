from typing import Any, Callable, Dict
from benchedge.metrics import BenchmarkResult


def run_backend_benchmark(
    backend: str,
    prompt_tokens: int,
    generated_tokens: int,
    trace_fn: Callable[[str, int, int], Dict[str, Any]],
) -> BenchmarkResult:
    raise NotImplementedError
