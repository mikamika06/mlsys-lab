import ref

def check(workdir):
    from loss_spike import AllReduceSimulator
    m = {"op_found": 0.0}
    r = AllReduceSimulator(64)
    res = r.reduce([1.0, 1.0, 1.0, 1.0])
    if isinstance(res, float):
        m["op_found"] = 1.0
    return m
