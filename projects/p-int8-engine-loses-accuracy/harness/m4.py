def check(workdir):
    m = {"calibration_ok": 0.0}
    try:
        from int8_eng.tuning import calibrate
        model = {"calibrated": False}
        res = calibrate(model, [10.0, 20.0])
        if res.get("calibrated") is True:
            m["calibration_ok"] = 1.0
    except Exception:
        pass
    return m
