def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from quant import calibration
    import ref
    m = {"bounded_drop": 0.0}
    data = ref.generate_synthetic_data()
    try:
        drops = calibration.evaluate_drop(data)
        if isinstance(drops, dict) and all(v < 0.05 for v in drops.values()):
            m["bounded_drop"] = 1.0
    except Exception:
        pass
    return m
