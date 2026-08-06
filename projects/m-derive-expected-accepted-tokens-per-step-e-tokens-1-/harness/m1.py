import ref

def check(workdir):
    from speculative.theory import expected_accepted_tokens
    test_cases = [(0.5, 3), (0.9, 4), (1.0, 5), (0.2, 2), (0.75, 7)]
    max_err = 0.0
    for a, g in test_cases:
        want = ref.expected_tokens(a, g)
        got = expected_accepted_tokens(a, g)
        err = abs(got - want) / (abs(want) + 1e-12)
        if err > max_err:
            max_err = err
    return {"rel_err": float(max_err)}
