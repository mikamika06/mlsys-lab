import pytest
from calib.hessian import estimate_hessian_memory

def test_hessian_memory_bounds():
    mem = estimate_hessian_memory(hidden_size=4096, num_parameters=1000000, dtype_bytes=4)
    assert mem > 0
    expected = (1000000 * 4) + (4096 * 4096 * 4)
    assert mem == expected
