def test_optimal_draft():
    from draftopt.opt import optimal_draft_length
    k = optimal_draft_length(0.8, 0.2)
    assert isinstance(k, int)
    assert k > 0
