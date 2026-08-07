def check(workdir):
    from scaler.policy import should_admit
    m = {"admission_ok": 0.0}
    try:
        r1 = should_admit(2, 10, 0.5)
        r2 = should_admit(12, 10, 0.5)
        if r1 is True and r2 is False:
            m["admission_ok"] = 1.0
    except Exception:
        pass
    return m
