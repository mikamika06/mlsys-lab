import sys

sys.path.insert(0, ".")
from tpval.feasibility import validate_tp_feasibility
from tpval.traffic import compute_pp_bubble_fraction, compute_tp_traffic

CONFIG = {
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 11008,
    "hidden_size": 4096,
    "num_layers": 32,
    "dtype_bytes": 2
}


def test_tp_feasibility_validation():
    res_valid = validate_tp_feasibility(CONFIG, 8)
    assert res_valid["is_feasible"] is True
    assert len(res_valid["reasons"]) == 0

    res_invalid = validate_tp_feasibility(CONFIG, 16)
    assert res_invalid["is_feasible"] is False
    assert len(res_invalid["reasons"]) > 0


def test_tp_traffic_calculation():
    traffic_tp1 = compute_tp_traffic(CONFIG, 1, 1000.0)
    assert traffic_tp1["bytes_per_token_per_rank"] == 0.0
    assert traffic_tp1["total_bus_bytes_per_sec"] == 0.0

    traffic_tp8 = compute_tp_traffic(CONFIG, 8, 1000.0)
    assert traffic_tp8["bytes_per_token_per_rank"] > 0.0
    assert traffic_tp8["total_bus_bytes_per_sec"] > 0.0


def test_pp_bubble_fraction():
    bubble_m8_p4 = compute_pp_bubble_fraction(8, 4)
    assert 0.0 < bubble_m8_p4 < 1.0
    assert abs(bubble_m8_p4 - 3.0 / 11.0) < 1e-6
