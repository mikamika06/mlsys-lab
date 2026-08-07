import sys
import os
import numpy as np
import ref

def check(workdir):
    m = {"teacher_eval_ok": 0.0}
    sys.path.insert(0, workdir)
    try:
        import distill.baseline as b
        x = np.ones((2, 2))
        w = np.ones((2, 2))
        out = b.evaluate_teacher(x, w)
        if out is not None:
            m["teacher_eval_ok"] = 1.0
    except Exception:
        pass
    return m
