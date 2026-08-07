import ref


def check(workdir):
    from leakdiag.loss import measure_loss_memory
    res = measure_loss_memory(100)
    out = {"leak_detected": 0.0}
    if res["ratio"] > 1.5:
        out["leak_detected"] = 1.0
    else:
        out["_note"] = f"Expected ratio > 1.5, got {res['ratio']}"
    return out
