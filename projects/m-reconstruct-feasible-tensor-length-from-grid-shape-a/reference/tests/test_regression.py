import sys

sys.path.insert(0, ".")
from tensorgrid.reconstruct import reconstruct_length
from tensorgrid.validate import find_optimal_grid, is_feasible


def test_reconstruct_bounds():
    min_l, max_l = reconstruct_length((3,), 128)
    assert min_l == 257
    assert max_l == 384


def test_feasibility():
    assert is_feasible(300, (3,), 128)
    assert not is_feasible(400, (3,), 128)


def test_optimal_grid():
    assert find_optimal_grid(300, 128) == 3
