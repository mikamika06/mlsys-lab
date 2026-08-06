import pytest
from accum.scaling import compute_gradient_inflation_factor

def test_gradient_inflation_scaling():
    factor = compute_gradient_inflation_factor(4, False)
    assert factor == 4.0
    factor_normalized = compute_gradient_inflation_factor(4, True)
    assert factor_normalized == 1.0
