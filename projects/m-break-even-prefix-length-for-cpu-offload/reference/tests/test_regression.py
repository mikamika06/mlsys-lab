from offload.simulator import compute_breakeven_prefix_length


def test_breakeven_positive_overhead():
    config = {
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "dtype_bytes": 2,
        "num_params": 7e9
    }
    hw = {"gpu_tflops": 100.0, "launch_overhead_s": 0.001}
    tier = {"bandwidth_gbps": 10.0, "latency_s": 0.05}
    be = compute_breakeven_prefix_length(config, hw, tier)
    assert be > 0.0
    assert be != float("inf")
