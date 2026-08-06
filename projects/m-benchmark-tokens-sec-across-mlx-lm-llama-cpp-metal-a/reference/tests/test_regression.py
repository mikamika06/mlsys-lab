from benchedge.metrics import compute_benchmark_metrics


def test_decode_duration_excludes_ttft():
    res = compute_benchmark_metrics(
        backend="mlx-lm",
        prompt_tokens=100,
        generated_tokens=31,
        t_start=10.0,
        t_first_token=12.0,
        t_end=15.0,
        rss_samples=[1000.0, 1024.0],
    )
    assert res.ttft_sec == 2.0
    assert res.decode_duration_sec == 3.0
    assert res.decode_tokens_per_sec == 10.0
