from capacity.batching import find_optimal_batch_size


def test_sla_batch_selection():
    profiles = [
        {"batch_size": 1, "latency_ms": 10.0, "tokens_per_sec": 100.0},
        {"batch_size": 2, "latency_ms": 15.0, "tokens_per_sec": 250.0},
        {"batch_size": 4, "latency_ms": 35.0, "tokens_per_sec": 600.0},
        {"batch_size": 8, "latency_ms": 80.0, "tokens_per_sec": 1000.0},
    ]
    res = find_optimal_batch_size(profiles, sla_latency_ms=20.0, cost_per_node_hr=3.6)
    assert res["optimal_batch_size"] == 2
