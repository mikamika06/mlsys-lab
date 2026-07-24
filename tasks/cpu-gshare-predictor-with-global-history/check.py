def grade(sol, fx) -> dict:
    # Two test cases: simple alternating pattern and longer random-like trace
    test_cases = [
        ([0x100, 0x104, 0x100, 0x108], [1, 0, 1, 1], 3),
        ([0x200, 0x204, 0x208, 0x20C], [1, 1, 1, 1], 4),
        ([0x100, 0x100, 0x100, 0x100], [1, 0, 1, 0], 4),
    ]
    try:
        for pc_list, outcome_list, k in test_cases:
            expected = _ref_simulate(pc_list, outcome_list, k)
            got = sol.simulate_branch(pc_list, outcome_list, k)
            if got != expected:
                return {"exact_match": 0.0}
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0}

def _ref_simulate(pc_list, outcome_list, k):
    size = 1 << k
    pht = [2] * size
    ghr = 0
    mispredictions = 0
    for pc, outcome in zip(pc_list, outcome_list):
        index = (pc ^ ghr) & (size - 1)
        prediction = 1 if pht[index] >= 2 else 0
        if prediction != outcome:
            mispredictions += 1
        if outcome:
            pht[index] = min(pht[index] + 1, 3)
        else:
            pht[index] = max(pht[index] - 1, 0)
        ghr = ((ghr << 1) | outcome) & (size - 1)
    return mispredictions
