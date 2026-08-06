from runner.capacity import select_optimal_num_parallel


def test_capacity_selection():
    benchmarks = [
        {"num_parallel": 1, "latencies_ms": [50, 60, 70, 80, 90], "duration_s": 10.0},
        {"num_parallel": 2, "latencies_ms": [100, 120, 130, 140, 300], "duration_s": 10.0},
        {"num_parallel": 4, "latencies_ms": [200, 250, 280, 310, 500], "duration_s": 10.0},
    ]
    res = select_optimal_num_parallel(benchmarks, p95_slo_ms=200.0)
    assert res["num_parallel"] == 1
    assert res["max_goodput"] > 0
