def check(workdir):
    from model.net import verify_warmup
    m = {"warmup_passed": 0.0}
    try:
        if verify_warmup():
            m["warmup_passed"] = 1.0
    except Exception:
        pass
    return m
