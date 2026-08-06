from profiling.derivation import derive_ai_formula
from profiling.classify import classify_phase_bound
from profiling.bandwidth import measure_decode_bandwidth

def test_derivation_scaling():
    res = derive_ai_formula(7e9, 4096, 32, 14e9, 128)
    assert res["prefill_flops_per_token"] > 0
    assert res["decode_bytes_per_token"] > 14e9

def test_classification_bound():
    cfg = {
        "params": 7e9,
        "hidden_size": 4096,
        "num_layers": 32,
        "total_weight_bytes": 14e9,
        "time_prefill_ms": 20.0,
        "time_decode_ms_per_token": 15.0
    }
    res = classify_phase_bound(cfg, "decode", 10, 1e12, 150.0)
    assert res["bound"] in ("memory", "compute")

def test_bandwidth_measurement():
    bw = measure_decode_bandwidth(14e9, 20.0)
    assert bw > 0.0
