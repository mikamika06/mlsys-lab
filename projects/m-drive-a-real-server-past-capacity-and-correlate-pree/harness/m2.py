import ref
import numpy as np

def check(workdir):
    from serverload.correlate import correlate
    out = {"rel_err": 1.0}
    errs = []
    for sc in ref.SCENARIOS:
        res = ref.simulate_load(sc)
        want = ref.correlate(res["latencies"], res["preemptions"])
        got = correlate(res["latencies"], res["preemptions"])
        err = abs(want - got) / (abs(want) + 1e-8)
        errs.append(err)
    max_err = float(max(errs)) if errs else 1.0
    out["rel_err"] = max_err
    return out
