import ref

def check(workdir):
    from keepalive.metrics import estimate_load_duration
    test_cases = [
        (1000.0, "warm"),
        (2048.0, "page_cached"),
        (4096.0, "cold"),
        (512.0, "page_cached"),
        (10240.0, "cold")
    ]
    max_err = 0.0
    for size, state in test_cases:
        want = ref.estimate_load_duration(size, state)
        got = estimate_load_duration(size, state)
        if want == 0.0:
            err = abs(got - want)
        else:
            err = abs(got - want) / abs(want)
        if err > max_err:
            max_err = err
    return {"rel_err": float(max_err)}
