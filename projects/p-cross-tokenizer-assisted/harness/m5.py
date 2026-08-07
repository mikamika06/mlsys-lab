def check(workdir):
    from speculative.metrics import compute_speedup
    m = {"speedup_achieved": 0.0}
    try:
        val = compute_speedup(2.0, 1.0)
        if abs(val - 2.0) < 1e-5:
            m["speedup_achieved"] = 1.0
    except Exception:
        pass
    return m
