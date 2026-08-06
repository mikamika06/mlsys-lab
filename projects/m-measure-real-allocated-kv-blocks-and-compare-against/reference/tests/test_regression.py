import sys
sys.path.insert(0, ".")
from kvalloc.blocks import compute_budget
from kvalloc.metrics import compute_relative_error
from kvalloc.simulator import measure_allocated_blocks

def test_budget_basic():
    assert compute_budget([100, 200], 16, 2) == ((100 + 15) // 16 + (200 + 15) // 16) * 2

def test_relative_error_exact():
    assert compute_relative_error(100, 100) == 0.0

def test_simulator_unique():
    tables = [[0, 1, 2], [2, 3]]
    assert measure_allocated_blocks(tables, 16) == 4
