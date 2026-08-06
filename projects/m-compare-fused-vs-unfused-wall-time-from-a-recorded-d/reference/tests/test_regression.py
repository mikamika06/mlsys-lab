from bench_analysis.report import generate_benchmark_summary


def test_metric_invariants():
    record = {
        "shape": [1024, 1024],
        "dtype": "float32",
        "num_inputs": 2,
        "num_outputs": 1,
        "num_unfused_ops": 2,
        "unfused_trace": {"samples_ms": [2.0, 2.0, 2.0]},
        "fused_trace": {"samples_ms": [1.0, 1.0, 1.0]},
    }
    summary = generate_benchmark_summary(record)
    assert abs(summary["speedup"] - 2.0) < 1e-5
    assert summary["fused_mean_ms"] < summary["unfused_mean_ms"]
    assert summary["fused_gbps"] > 0
    assert summary["unfused_gbps"] > 0
