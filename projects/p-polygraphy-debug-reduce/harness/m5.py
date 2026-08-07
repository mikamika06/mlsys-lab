import ref

def check(workdir):
    m = {"parity_ok": 0.0}
    try:
        ok = ref.oracle_verify()
        if ok:
            m["parity_ok"] = 1.0
    except Exception:
        pass
    return m
