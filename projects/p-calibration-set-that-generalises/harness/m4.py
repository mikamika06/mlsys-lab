def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from quant import calibration
    import ref
    m = {"three_domains_checked": 0.0}
    data = ref.generate_synthetic_data()
    try:
        res = calibration.check_domains(data)
        if isinstance(res, dict) and len(res) >= 3:
            m["three_domains_checked"] = 1.0
    except Exception:
        pass
    return m
