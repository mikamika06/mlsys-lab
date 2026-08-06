from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkResult:
    backend: str
    prompt_tokens: int
    generated_tokens: int
    ttft_sec: float
    decode_duration_sec: float
    decode_tokens_per_sec: float
    peak_rss_mb: float


def compute_benchmark_metrics(
    backend: str,
    prompt_tokens: int,
    generated_tokens: int,
    t_start: float,
    t_first_token: float,
    t_end: float,
    rss_samples: List[float],
) -> BenchmarkResult:
    raise NotImplementedError
