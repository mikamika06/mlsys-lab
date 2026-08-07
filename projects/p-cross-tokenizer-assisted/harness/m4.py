def check(workdir):
    from speculative.metrics import measure_acceptance
    m = {"acceptance_measured": 0.0}
    try:
        val = measure_acceptance(8, 10)
        if abs(val - 0.8) < 1e-5:
            m["acceptance_measured"] = 1.0
    except Exception:
        pass
    return m
