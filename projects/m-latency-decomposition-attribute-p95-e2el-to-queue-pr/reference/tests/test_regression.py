import sys

sys.path.insert(0, ".")
from latmetrics.decomposition import calculate_percentile, decompose_latencies
from latmetrics.goodput import evaluate_slo, rank_configs


def test_percentile_methods():
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    near = calculate_percentile(data, 50.0, method="nearest")
    lin = calculate_percentile(data, 50.0, method="linear")
    assert near == 30.0
    assert lin == 30.0


def test_latency_decomposition_shares():
    reqs = [
        {"queue_ms": 10.0, "prefill_ms": 20.0, "decode_ms": 70.0},
        {"queue_ms": 20.0, "prefill_ms": 30.0, "decode_ms": 50.0},
    ]
    res = decompose_latencies(reqs, method="nearest")
    share_sum = res["queue_share"] + res["prefill_share"] + res["decode_share"]
    assert abs(share_sum - 1.0) < 1e-6


def test_rank_configs_prioritizes_goodput_over_throughput():
    cfg_a = {
        "config_id": "high_tp_low_gp",
        "duration_s": 1.0,
        "requests": [
            {"ttft_ms": 500.0, "tpot_ms": 10.0, "output_tokens": 1000}
        ],
    }
    cfg_b = {
        "config_id": "low_tp_high_gp",
        "duration_s": 1.0,
        "requests": [
            {"ttft_ms": 50.0, "tpot_ms": 2.0, "output_tokens": 500}
        ],
    }
    ranked = rank_configs([cfg_a, cfg_b], slo_ttft_ms=100.0, slo_tpot_ms=5.0)
    assert ranked[0]["config_id"] == "low_tp_high_gp"
    assert ranked[1]["config_id"] == "high_tp_low_gp"
