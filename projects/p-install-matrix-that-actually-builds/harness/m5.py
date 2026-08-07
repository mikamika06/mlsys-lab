def check(workdir):
    from install.builder import verify_fresh_install
    m = {"reproducible_zero": 0.0}
    try:
        ok = verify_fresh_install()
        if ok is True:
            m["reproducible_zero"] = 1.0
    except Exception:
        pass
    return m
