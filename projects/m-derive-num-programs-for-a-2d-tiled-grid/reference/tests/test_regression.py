import pytest
from triton_grid.config import derive_num_programs

def test_grid_boundary_coverage():
    width, height = 100, 100
    block_w, block_h = 32, 32
    gx, gy, total = derive_num_programs(width, height, block_w, block_h)
    assert gx == 4
    assert gy == 4
    assert total == 16
