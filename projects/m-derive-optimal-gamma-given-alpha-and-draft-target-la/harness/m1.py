import ref


def check(workdir):
    from speculative.optimal import derive_optimal_gamma

    matched = 0
    for case in ref.TEST_CASES:
        alpha = case["alpha"]
        c = case["c"]
        mg = case["max_gamma"]
        want = ref.compute_optimal_gamma(alpha, c, mg)
        got = derive_optimal_gamma(alpha, c, mg)
        if got == want:
            matched += 1
    return {"derivations_matched": float(matched)}
