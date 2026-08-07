def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from quant import calibration
    import ref
    m = {"min_size_found": 0.0}
    data = ref.generate_synthetic_data()
    try:
        sz = calibration.find_min_size(data)
        if isinstance(sz, int) and sz > 0:
            m["min_size_found"] = 1.0
    except Exception:
        pass
    return m
