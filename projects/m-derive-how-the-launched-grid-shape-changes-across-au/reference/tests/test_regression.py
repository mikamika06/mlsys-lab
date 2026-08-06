import pytest
from grid_analyzer.core import derive_grid

def test_grid_bounds():
    cfg = {"BLOCK_M": 32, "BLOCK_N": 32}
    g = derive_grid(100, 100, cfg)
    assert g == (4, 4)

def test_exact_fit():
    cfg = {"BLOCK_M": 16, "BLOCK_N": 16}
    g = derive_grid(32, 32, cfg)
    assert g == (2, 2)
