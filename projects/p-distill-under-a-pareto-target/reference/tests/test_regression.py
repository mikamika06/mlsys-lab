import sys
sys.path.insert(0, ".")
from distill.pareto import check_pareto

def test_pareto_boundary():
    assert check_pareto(0.95, 0.939, 100, 50) is True
    assert check_pareto(0.95, 0.92, 100, 50) is False
