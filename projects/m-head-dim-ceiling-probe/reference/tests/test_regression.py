import sys
sys.path.insert(0, ".")
from probe.head_dim import check_head_ceiling
from probe.throughput import estimate_throughput
from probe.fp8_gating import check_fp8_availability


def test_head_ceiling_bounds():
    cfg = {"head_dim": 256}
    res = check_head_ceiling(cfg)
    assert res["max_supported_dim"] >= 256
    assert isinstance(res["fa3_supported"], bool)


def test_throughput_comparison():
    cfg = {"head_dim": 128, "hardware": "hopper"}
    t = estimate_throughput(cfg)
    assert "fa2" in t and "fa3" in t
    assert t["fa3"] > 0


def test_fp8_gating_logic():
    cfg = {"head_dim": 256, "dtype": "fp8"}
    res = check_fp8_availability(cfg)
    assert res["available"] is False
    assert res["reason"] == "head_dim_exceeds_fp8_limit"
