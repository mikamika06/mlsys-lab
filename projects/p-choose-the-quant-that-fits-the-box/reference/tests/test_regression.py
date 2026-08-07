import sys
sys.path.insert(0, ".")
from quant.analyzer import compute_bpw_and_size, measure_peak_memory, measure_quality, measure_speed
from quant.selector import generate_recommendation_table, auto_select_recipe
import numpy as np

def test_bpw_accuracy():
    bpw, size = compute_bpw_and_size(7000000000, 4.5)
    assert bpw == 4.5
    assert size == int(7000000000 * 4.5 / 8.0)

def test_peak_memory():
    mem = measure_peak_memory(7000000000, 4.0, 2048.0)
    expected = (7000000000 * 4.0 / 8.0) / (1024 * 1024) + 2048.0
    assert abs(mem - expected) < 1e-5

def test_quality_identical():
    logits = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    res = measure_quality(logits, logits)
    assert res["kld"] < 1e-5

def test_speed_scaling():
    s1 = measure_speed(4.0, 100.0)
    s2 = measure_speed(8.0, 100.0)
    assert s1 > s2

def test_selector_table():
    configs = [
        {"name": "q4", "bpw": 4.0, "peak_memory_mb": 5000.0},
        {"name": "q8", "bpw": 8.0, "peak_memory_mb": 9000.0}
    ]
    limits = [6.0, 10.0]
    tbl = generate_recommendation_table(configs, limits)
    assert tbl[0]["recommended_recipe"] == "q4"
    assert tbl[1]["recommended_recipe"] == "q8"

def test_auto_select():
    table = [
        {"memory_limit_gb": 16.0, "recommended_recipe": "q4"},
        {"memory_limit_gb": 36.0, "recommended_recipe": "q8"}
    ]
    assert auto_select_recipe(20.0, table) == "q4"
    assert auto_select_recipe(40.0, table) == "q8"
