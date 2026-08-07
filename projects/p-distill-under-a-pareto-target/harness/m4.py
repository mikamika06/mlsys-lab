import sys
import numpy as np
import ref

def check(workdir):
    m = {"hyperparam_tune_ok": 0.0}
    sys.path.insert(0, workdir)
    try:
        import distill.tune as t
        res = t.find_best_hyperparams([0.5], [2.0], [0.9])
        if res is not None:
            m["hyperparam_tune_ok"] = 1.0
    except Exception:
        pass
    return m
