from batching.waste import compute_padding_waste
from batching.simulator import simulate_batcher


def test_batching_waste_and_throughput():
    requests = [
        {"id": 1, "arrival_time": 0, "prompt_len": 10, "decode_len": 5},
        {"id": 2, "arrival_time": 0, "prompt_len": 50, "decode_len": 20},
        {"id": 3, "arrival_time": 5, "prompt_len": 12, "decode_len": 8},
        {"id": 4, "arrival_time": 5, "prompt_len": 30, "decode_len": 15},
    ]

    waste = compute_padding_waste(requests, max_batch_size=2)
    assert waste["static_padded_tokens"] > 0
    assert waste["static_waste_ratio"] > waste["continuous_waste_ratio"]

    sim_static = simulate_batcher(requests, max_batch_size=2, mode="static", step_time_ms=10.0)
    sim_continuous = simulate_batcher(requests, max_batch_size=2, mode="continuous", step_time_ms=10.0)

    assert sim_continuous["throughput_tokens_per_sec"] >= sim_static["throughput_tokens_per_sec"]
    assert sim_continuous["avg_latency_ms"] <= sim_static["avg_latency_ms"]
