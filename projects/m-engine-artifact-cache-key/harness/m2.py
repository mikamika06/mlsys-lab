import ref

def check(workdir):
    from engcache.batching import optimal_queue_delay
    test_cases = [
        (150.0, 600.0, 16, 0.04),
        (200.0, 1000.0, 32, 0.02),
        (50.0, 300.0, 8, 0.05)
    ]
    max_err = 0.0
    for arrival, service, max_bs, target in test_cases:
        want = ref.optimal_queue_delay(arrival, service, max_bs, target)
        got = optimal_queue_delay(arrival, service, max_bs, target)
        err = abs(got - want) / (abs(want) + 1e-9)
        if err > max_err:
            max_err = err
    return {"rel_err": float(max_err)}
