def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from quant.sensitivity import measure_sensitivity
    from harness.ref import get_dummy_data

    model, dl = get_dummy_data()
    m = {"sensitivity_computed": 0.0}
    try:
        sens = measure_sensitivity(model, dl)
        if isinstance(sens, dict) and len(sens) == len(model):
            m["sensitivity_computed"] = 1.0
    except Exception:
        pass
    return m
