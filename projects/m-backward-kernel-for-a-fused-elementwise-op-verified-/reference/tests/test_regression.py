import numpy as np
from fused_grad.atomic_analysis import classify_atomic_requirement
from fused_grad.backward import finite_difference_grad_x, fused_elementwise_backward


def test_atomic_requirement_on_overlapping_map():
    overlapping_map = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    assert classify_atomic_requirement(overlapping_map) is True

    unique_map = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    assert classify_atomic_requirement(unique_map) is False


def test_backward_matches_finite_differences():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(6)
    index_map = np.array([0, 2, 1, 2, 5, 0], dtype=np.int64)
    grad_output = rng.standard_normal(len(index_map))

    analytic = fused_elementwise_backward(grad_output, x, index_map, use_atomic=True)
    numeric = finite_difference_grad_x(x, index_map, grad_output, eps=1e-5)

    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-4)
