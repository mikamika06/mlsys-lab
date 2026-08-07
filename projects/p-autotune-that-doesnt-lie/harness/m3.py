def check(workdir):
    from autotune.metrics import measure_latency
    m = {"warmup_and_reps_ok": 0.0}
    val = measure_latency(lambda: 1 + 1, [], warmup=5, reps=10)
    if isinstance(val, (int, float)) and val >= 0:
        m["warmup_and_reps_ok"] = 1.0
    return m
