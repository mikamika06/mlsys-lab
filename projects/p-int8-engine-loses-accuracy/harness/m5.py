def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from engine.quantize import verify_accuracy_and_speedup
    from harness.ref import generate_reference_model, get_calibration_set

    m = {"accuracy_recovered": 0.0, "speedup_retained": 0.0}
    model = generate_reference_model(12)
    calib = get_calibration_set()

    try:
        acc_ok, speed_ok = verify_accuracy_and_speedup(model, calib)
    except Exception:
        return m

    if acc_ok:
        m["accuracy_recovered"] = 1.0
    if speed_ok:
        m["speedup_retained"] = 1.0
    return m
