from benchedge.metrics import compute_benchmark_metrics

TRACES = {
    "mlx-lm": {
        "prompt_tokens": 512,
        "generated_tokens": 128,
        "t_start": 100.0,
        "t_first_token": 100.12,
        "t_end": 102.66,
        "rss_samples": [2048.0, 3100.0, 3150.0, 3120.0],
    },
    "llama.cpp-metal": {
        "prompt_tokens": 512,
        "generated_tokens": 128,
        "t_start": 200.0,
        "t_first_token": 200.08,
        "t_end": 202.16,
        "rss_samples": [1800.0, 2600.0, 2650.0, 2640.0],
    },
    "torch-mps": {
        "prompt_tokens": 512,
        "generated_tokens": 128,
        "t_start": 300.0,
        "t_first_token": 300.25,
        "t_end": 304.45,
        "rss_samples": [2200.0, 3800.0, 3900.0, 3850.0],
    },
}


def mock_trace_fn(backend, prompt_tokens, generated_tokens):
    return TRACES[backend]


def compute_ref_metrics(backend, trace):
    return compute_benchmark_metrics(
        backend=backend,
        prompt_tokens=trace["prompt_tokens"],
        generated_tokens=trace["generated_tokens"],
        t_start=trace["t_start"],
        t_first_token=trace["t_first_token"],
        t_end=trace["t_end"],
        rss_samples=trace["rss_samples"],
    )


def build_ref_summary(results, baseline_backend="torch-mps"):
    by_backend = {r.backend: r for r in results}
    base = by_backend[baseline_backend]
    summary = {}
    for backend, res in sorted(by_backend.items()):
        tps_ratio = (
            res.decode_tokens_per_sec / base.decode_tokens_per_sec
            if base.decode_tokens_per_sec > 0
            else 0.0
        )
        ttft_ratio = res.ttft_sec / base.ttft_sec if base.ttft_sec > 0 else 0.0
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
