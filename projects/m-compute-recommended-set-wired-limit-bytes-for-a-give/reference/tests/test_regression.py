def test_tune_limits_prevents_oom():
    from limits.compute import tune_limits
    gb = 1024 ** 3
    memsize = 32 * gb
    model_bytes = 20 * gb

    wired, cache = tune_limits(memsize, model_bytes)
    assert cache + model_bytes <= wired
