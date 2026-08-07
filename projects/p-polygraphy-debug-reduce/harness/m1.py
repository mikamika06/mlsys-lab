import ref

def check(workdir):
    m = {"step_compare_ok": 0.0}
    try:
        res = ref.oracle_step_compare()
        if "layer1" in res:
            m["step_compare_ok"] = 1.0
    except Exception:
        pass
    return m
