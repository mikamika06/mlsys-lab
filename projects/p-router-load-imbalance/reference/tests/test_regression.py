import numpy as np
from moe.metrics import compute_imbalance_ratio, measure_distribution
from moe.router import MoERouter, simulate_step_time


def test_distribution_sum_matches_tokens():
    selected = np.array([[0, 1], [1, 2], [0, 3]])
    counts = measure_distribution(selected, 4)
    assert np.sum(counts) == selected.size


def test_imbalance_ratio_bounds():
    counts = np.array([10, 10, 10, 10])
    ratio = compute_imbalance_ratio(counts)
    assert abs(ratio - 1.0) < 1e-5


def test_router_outputs_valid_probabilities():
    router = MoERouter(num_experts=4, in_dim=8)
    x = np.random.randn(10, 8)
    probs, idxs, weights = router.route(x, top_k=2)
    assert np.allclose(np.sum(probs, axis=-1), 1.0)
    assert idxs.shape == (10, 2)


def test_step_time_increases_with_imbalance():
    balanced = np.array([50, 50, 50, 50])
    imbalanced = np.array([140, 20, 20, 20])
    t1 = simulate_step_time(balanced)
    t2 = simulate_step_time(imbalanced)
    assert t2 > t1
