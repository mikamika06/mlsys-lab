import numpy as np
from vadd.grid import calculate_launch_waste, get_grid_num_programs
from vadd.kernel import run_vector_add


def test_grid_math_and_launch_waste():
    assert get_grid_num_programs(1000, 128) == 8
    assert calculate_launch_waste(1000, 128) == 24
    assert get_grid_num_programs(1024, 128) == 8
    assert calculate_launch_waste(1024, 128) == 0


def test_correct_grid_computes_all_elements():
    np.random.seed(42)
    x = np.random.randn(1000).astype(np.float32)
    y = np.random.randn(1000).astype(np.float32)
    out, dropped = run_vector_add(x, y, block_size=128, grid_type="correct")
    assert dropped == 0
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out, x + y, rtol=1e-5)


def test_underlaunched_grid_drops_tail_elements():
    np.random.seed(42)
    x = np.random.randn(1000).astype(np.float32)
    y = np.random.randn(1000).astype(np.float32)
    out, dropped = run_vector_add(x, y, block_size=128, grid_type="underlaunched")
    assert dropped == 104
    assert np.isnan(out[896:]).all()
    np.testing.assert_allclose(out[:896], x[:896] + y[:896], rtol=1e-5)
