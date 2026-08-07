def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from quant import calibration
    import ref
    m = {"sensitivity_measured": 0.0}
    data = ref.generate_synthetic_data()
    try:
        res = calibration.measure_sensitivity(data)
        if isinstance(res, dict) and len(res) > 0:
            m["sensitivity_measured"] = 1.0
    except Exception:
        pass
    return m
