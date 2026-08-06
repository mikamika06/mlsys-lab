from ddpplan.divergence import check_rank_plan_consistency


def test_consistent_plans():
    plan = [["layer2.weight"], ["layer1.weight"]]
    rank_plans = {0: plan, 1: plan, 2: plan}
    res = check_rank_plan_consistency(rank_plans)
    assert res["consistent"] is True
    assert res["mismatched_ranks"] == []


def test_divergent_plans():
    plan_a = [["layer2.weight"], ["layer1.weight"]]
    plan_b = [["layer2.weight", "layer1.weight"]]
    rank_plans = {0: plan_a, 1: plan_b, 2: plan_a}
    res = check_rank_plan_consistency(rank_plans)
    assert res["consistent"] is False
    assert 1 in res["mismatched_ranks"]
