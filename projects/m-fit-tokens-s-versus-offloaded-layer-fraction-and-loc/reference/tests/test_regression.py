from offload.memory import fit_layers_in_budget
from offload.planner import select_optimal_offload
from offload.profiler import find_offload_cliff

CONFIG = {
    "total_layers": 32,
    "base_overhead_bytes": 1024 * 1024 * 500,
    "bytes_per_layer_weight": 1024 * 1024 * 100,
    "bytes_per_layer_kv": 1024 * 1024 * 20,
}

PROFILES = {
    0.0: 80.0,
    0.25: 75.0,
    0.5: 70.0,
    0.75: 20.0,
    1.0: 5.0,
}


def test_fit_layers_monotonicity():
    b1 = 1024 * 1024 * 1000
    b2 = 1024 * 1024 * 2000
    l1 = fit_layers_in_budget(CONFIG, b1)
    l2 = fit_layers_in_budget(CONFIG, b2)
    assert l2 >= l1
    assert l1 <= CONFIG["total_layers"]
    assert l2 <= CONFIG["total_layers"]


def test_fit_layers_zero_budget():
    assert fit_layers_in_budget(CONFIG, 0) == 0


def test_planner_respects_max_budget():
    budget = 1024 * 1024 * 1500
    max_fit = fit_layers_in_budget(CONFIG, budget)
    selected_gpu = select_optimal_offload(CONFIG, budget, PROFILES)
    assert selected_gpu <= max_fit


def test_cliff_detection_location():
    cliff = find_offload_cliff(PROFILES)
    assert cliff == 0.75
