import ref
import numpy as np

def check(workdir):
    from tpscaling.table import compute_scaling_table
    out = {"max_rel_err": 0.0}
    max_err = 0.0
    tp_degrees = [1, 2, 4, 8]
    for cfg in ref.CONFIGS:
        want = ref.compute_scaling_table(cfg, tp_degrees)
        got = compute_scaling_table(cfg, tp_degrees)
        for w, g in zip(want, got):
            for k in ["memory_gb", "comm_bytes"]:
                err = abs(g[k] - w[k]) / (abs(w[k]) + 1e-9)
                if err > max_err:
                    max_err = err
    out["max_rel_err"] = float(max_err)
    return out
