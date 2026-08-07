import sys
sys.path.insert(0, ".")
from quantplan.pareto import compute_pareto
from quantplan.budget import best_config
from quantplan.footprint import total_footprint


def test_pareto_non_empty():
    res = compute_pareto((256, 256), [32, 64], [False, True])
    assert len(res) == 4


def test_budget_selection():
    cfg = best_config((256, 256), [32, 64], [False, True], 0.002)
    assert "memory_bytes" in cfg


def test_footprint_includes_non_quantized():
    fp = total_footprint((256, 256), [32], [False], 5000, 32, False)
    assert fp > 5000
