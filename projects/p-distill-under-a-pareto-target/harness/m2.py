import sys
import numpy as np
import ref

def check(workdir):
    m = {"logit_distill_ok": 0.0}
    sys.path.insert(0, workdir)
    try:
        import distill.logits as l
        res = l.distill_logits(np.zeros((2, 2)), np.zeros((2, 2)))
        if isinstance(res, (float, np.floating)):
            m["logit_distill_ok"] = 1.0
    except Exception:
        pass
    return m
