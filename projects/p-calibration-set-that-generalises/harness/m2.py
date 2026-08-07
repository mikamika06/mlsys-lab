def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from quant import calibration
    import ref
    m = {"domains_compared": 0.0}
    data = ref.generate_synthetic_data()
    sens = ref.measure_sensitivity(data)
    try:
        diff = calibration.compare_domains(sens)
        if isinstance(diff, (int, float)):
            m["domains_compared"] = 1.0
    except Exception:
        pass
    return m
