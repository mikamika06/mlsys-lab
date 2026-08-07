import ref

def check(workdir):
    from ort_tune.optimizer import select_opt_level
    m = {"optimizer_level_ok": 0.0}
    lvl = select_opt_level([1, 2, 3], 90.0)
    if lvl in [1, 2, 99]:
        m["optimizer_level_ok"] = 1.0
    return m
