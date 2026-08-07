import ref

def check(workdir):
    m = {"bisection_ok": 0.0}
    try:
        idx = ref.oracle_bisection()
        if idx == 2:
            m["bisection_ok"] = 1.0
    except Exception:
        pass
    return m
