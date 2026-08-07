def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from engine.calibrate import calibrate_scales

    m = {"calibration_diverse": 0.0, "scales_optimized": 0.0}
    mock_dataset = [np.zeros((4, 4), dtype=np.float32), np.ones((4, 4), dtype=np.float32) * 5.0]

    try:
        scales = calibrate_scales(mock_dataset)
    except Exception:
        return m

    if isinstance(scales, (dict, list, float, np.ndarray)):
        m["calibration_diverse"] = 1.0
        if not (isinstance(scales, (int, float)) and scales == 0.0):
            m["scales_optimized"] = 1.0
    return m
