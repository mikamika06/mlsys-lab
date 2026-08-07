def check(workdir):
    from admit.policy import estimate_latency, should_admit
    m = {"admission_ok": 0.0}
    lat = estimate_latency(10, 5.0)
    if lat != 2.0:
        return m
    if not should_admit(1.5, 2.0, 0):
        return m
    if should_admit(2.5, 2.0, 0):
        return m
    m["admission_ok"] = 1.0
    return m
