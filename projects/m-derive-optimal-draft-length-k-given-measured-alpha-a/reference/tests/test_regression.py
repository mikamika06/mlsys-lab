def test_optimal_draft():
    from draftopt.opt import optimal_draft_length
    k = optimal_draft_length(0.8, 0.2)
    assert isinstance(k, int)
    assert k > 0
    assert optimal_draft_length(0.9, 0.1) >= optimal_draft_length(0.3, 0.4)

def test_tree_acceptance():
    from draftopt.tree import evaluate_tree_acceptance
    res = evaluate_tree_acceptance({"depth": 3, "branch_factor": 2, "paths": [[1, 1]]}, 2)
    assert "tree_accepted_tokens" in res
