import ref

def check(workdir):
    from loss_spike import AllReduceSimulator
    m = {"deterministic_ok": 0.0}
    r = AllReduceSimulator(64)
    vals = [1.0, 2.0, 3.0, 4.0]
    res1 = r.reduce(vals)
    res2 = r.reduce(vals)
    if res1 == res2:
        m["deterministic_ok"] = 1.0
    return m
