import ref

def check(workdir):
    from quantcal.loop import calibrate
    model = lambda x: x + 1
    data = [1, 2, 3, 4, 5]
    res = calibrate(model, data, 3)
    ok = 1 if len(res) == 3 else 0
    return {"peak_memory_cut": float(ok)}
