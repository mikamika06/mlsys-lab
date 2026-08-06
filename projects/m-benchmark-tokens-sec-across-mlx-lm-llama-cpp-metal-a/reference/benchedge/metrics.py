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
    ttft = max(0.0, t_first_token - t_start)
    decode_duration = max(0.0, t_end - t_first_token)
    if generated_tokens > 1 and decode_duration > 0:
        decode_tps = (generated_tokens - 1) / decode_duration
    else:
        decode_tps = 0.0
    peak_rss = max(rss_samples) if rss_samples else 0.0
    return BenchmarkResult(
        backend=backend,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        ttft_sec=round(ttft, 6),
        decode_duration_sec=round(decode_duration, 6),
        decode_tokens_per_sec=round(decode_tps, 4),
        peak_rss_mb=round(peak_rss, 2),
    )
