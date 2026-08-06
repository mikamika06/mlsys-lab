from vllm_cost.model import compute_breakeven_volume, compute_spot_expected_cost, compute_prefix_caching_savings


def test_breakeven_edge_cases():
    assert compute_breakeven_volume(1000, 0.001, 0.002) == float('inf')
    assert compute_breakeven_volume(1000, 0.002, 0.001) == 1000.0 / 0.001


def test_spot_cost_monotonicity():
    cost_low = compute_spot_expected_cost(2.0, 0.05, 0.1, 0.5, 100)
    cost_high = compute_spot_expected_cost(2.0, 0.20, 0.1, 0.5, 100)
    assert cost_high > cost_low


def test_prefix_caching_scaling():
    s1 = compute_prefix_caching_savings(1000, 500, 100, 0.5, 0.00001, 0.5)
    s2 = compute_prefix_caching_savings(2000, 500, 100, 0.5, 0.00001, 0.5)
    assert abs(s2 - 2.0 * s1) < 1e-9
