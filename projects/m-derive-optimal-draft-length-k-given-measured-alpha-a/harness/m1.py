import ref

def check(workdir):
    from draftopt.opt import optimal_draft_length
    cases = ref.get_test_cases()
    ok = 0
    for i, case in enumerate(cases):
        want = optimal_draft_length_ref(case["alpha"], case["cost_ratio"])
        got = optimal_draft_length(case["alpha"], case["cost_ratio"])
        if got == want:
            ok += 1
    out = {"optimal_k_matched": 1.0 if ok == len(cases) else 0.0}
    return out

def optimal_draft_length_ref(alpha, cost_ratio):
    best_k = 1
    best_speedup = 0.0
    for k in range(1, 32):
        expected_accepted = sum(alpha**i for i in range(1, k + 1))
        speedup = (1.0 + expected_accepted) / (1.0 + cost_ratio * k)
        if speedup > best_speedup:
            best_speedup = speedup
            best_k = k
    return best_k
