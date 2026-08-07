def check(workdir):
    from runner.core import correct_thermal

    m = {"thermal_accounting_ok": 0.0}
    try:
        res = correct_thermal([40, 38, 35], degradation_rate=0.05)
        if isinstance(res, list) and len(res) == 3:
            m["thermal_accounting_ok"] = 1.0
    except Exception:
        pass
    return m
