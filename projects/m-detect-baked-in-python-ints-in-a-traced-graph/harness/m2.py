import ref


def check(workdir):
    from graphguard.conditioner import rewrite_conditional
    def sample_fn(x):
        return x * 2 if x > 0 else x * -1

    fn = rewrite_conditional(sample_fn)
    test_vals = [-5, 0, 3, 12]
    match = 0
    try:
        for v in test_vals:
            if fn(v) == sample_fn(v):
                match += 1
    except Exception:
        pass
    return {"equivalence_match": 1.0 if match == len(test_vals) else 0.0}
