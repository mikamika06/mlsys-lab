from picker.payback import build_normalized_table, calculate_payback_volume
from picker.selector import select_backend


def test_payback_volume_calculation():
    val = calculate_payback_volume(30.0, 10.0, 5.0)
    assert val == 6000
    invalid = calculate_payback_volume(30.0, 5.0, 10.0)
    assert invalid == -1


def test_select_backend():
    res = select_backend({"device": "cpu"})
    assert res == "ort_cpu"


def test_normalized_table():
    candidates = [
        {"backend": "ort_cuda", "latency_ms": 10.0, "build_time_sec": 0.0},
        {"backend": "ort_trt", "latency_ms": 5.0, "build_time_sec": 30.0}
    ]
    table = build_normalized_table(candidates, baseline_backend="ort_cuda")
    assert len(table) == 2
    assert table[1]["payback_volume"] == 6000
