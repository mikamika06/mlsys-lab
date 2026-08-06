from typing import Any, Dict, List
from benchedge.metrics import BenchmarkResult


def summarize_benchmark_runs(
    results: List[BenchmarkResult], baseline_backend: str = "torch-mps"
) -> Dict[str, Any]:
    raise NotImplementedError
