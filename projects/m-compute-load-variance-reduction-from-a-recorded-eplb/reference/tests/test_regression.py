from eplb.bounds import min_redundant_replicas


def test_min_redundant_replicas_tight_bounds():
    expert_loads = [100.0, 10.0, 10.0, 10.0]
    num_ranks = 2
    target_max_load = 40.0

    req = min_redundant_replicas(expert_loads, num_ranks, target_max_load)

    assert req > 0
    assert req >= 2
