from ggufsplit.sharding import compute_split_plan_with_tensors


def test_tensor_limit_enforced():
    tensors = [("t1", 100), ("t2", 100), ("t3", 100)]
    plan = compute_split_plan_with_tensors(tensors, max_size=1000, max_tensors=2)
    assert len(plan) == 2
    assert len(plan[0]) == 2
    assert len(plan[1]) == 1
