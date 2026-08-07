def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from quant.analyzer import measure_speed
    import ref

    m = {"speed_measured": 0.0}
    try:
        val = measure_speed(4.0, 100.0)
        oval = ref.oracle_measure_speed(4.0, 100.0)
        if abs(val - oval) < 1e-5:
            m["speed_measured"] = 1.0
    except Exception:
        pass
    return m
