from typing import Any, Dict, List
from benchedge.metrics import BenchmarkResult


def summarize_benchmark_runs(
    results: List[BenchmarkResult], baseline_backend: str = "torch-mps"
) -> Dict[str, Any]:
    by_backend = {r.backend: r for r in results}
    if baseline_backend not in by_backend:
        raise ValueError(f"Baseline backend {baseline_backend} missing")

    base = by_backend[baseline_backend]
    summary = {}
    for backend, res in sorted(by_backend.items()):
        tps_ratio = (
            res.decode_tokens_per_sec / base.decode_tokens_per_sec
            if base.decode_tokens_per_sec > 0
            else 0.0
        )
        ttft_ratio = (
            res.ttft_sec / base.ttft_sec if base.ttft_sec > 0 else 0.0
        )
        rss_delta_mb = res.peak_rss_mb - base.peak_rss_mb
        summary[backend] = {
            "decode_tps": res.decode_tokens_per_sec,
            "ttft_sec": res.ttft_sec,
            "peak_rss_mb": res.peak_rss_mb,
            "throughput_ratio_vs_baseline": round(tps_ratio, 4),
            "ttft_ratio_vs_baseline": round(ttft_ratio, 4),
            "rss_delta_mb_vs_baseline": round(rss_delta_mb, 2),
        }
    return summary
