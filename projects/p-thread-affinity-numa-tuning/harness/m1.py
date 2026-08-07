import ref


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from numa_tuning import analyzer

    m = {"scaling_measured": 0.0, "efficiency_valid": 0.0}
    try:
        data = analyzer.measure_scaling([1, 2, 4], 100)
        if not isinstance(data, dict) or len(data) != 3:
            return m
        m["scaling_measured"] = 1.0

        eff = analyzer.calculate_efficiency(data[1], data[2], 2.0)
        if eff > 0.0:
            m["efficiency_valid"] = 1.0
    except Exception:
        pass
    return m
