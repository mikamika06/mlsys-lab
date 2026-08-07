def check(workdir):
    from speculative.transfer import transfer_candidates
    m = {"transfer_ok": 0.0}
    try:
        vocab = {65: 10, 66: 20}
        res = transfer_candidates([65, 99], vocab)
        if res == [10, 0]:
            m["transfer_ok"] = 1.0
    except Exception:
        pass
    return m
