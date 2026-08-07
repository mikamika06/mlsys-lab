def check(workdir):
    import sys
    import os
    sys.path.insert(0, workdir)
    from dl.profiler import measure_loader_fraction

    m = {"fraction_measured": 0.0}
    steps = [10.0, 10.0, 10.0]
    waits = [4.0, 4.0, 4.0]
    try:
        val = measure_loader_fraction(steps, waits)
        if abs(val - 0.4) < 1e-5:
            m["fraction_measured"] = 1.0
    except Exception:
        pass
    return m
