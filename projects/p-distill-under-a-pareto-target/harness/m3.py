import sys
import numpy as np
import ref

def check(workdir):
    m = {"hidden_distill_ok": 0.0}
    sys.path.insert(0, workdir)
    try:
        import distill.hidden as h
        res = h.distill_hidden(np.zeros((2, 2)), np.zeros((2, 2)))
        if isinstance(res, (float, np.floating)):
            m["hidden_distill_ok"] = 1.0
    except Exception:
        pass
    return m
