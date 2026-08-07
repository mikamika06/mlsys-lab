import ref
import numpy as np

def check(workdir):
    from loadtest.metrics import compute_metrics
    latencies = [10.0, 15.0, 20.0, 25.0, 30.0, 50.0, 100.0]
    duration = 2.0
    want = ref.compute_metrics(latencies, duration)
    got = compute_metrics(latencies, duration)

    rel_errs = []
    for k in ["p50", "p90", "p99", "throughput"]:
        w = want[k]
        g = got.get(k, 0.0)
        err = abs(g - w) / (abs(w) + 1e-9)
        rel_errs.append(err)
    max_err = float(max(rel_errs))
    out = {"rel_err": max_err}
    return out
