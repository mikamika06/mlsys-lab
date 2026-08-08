import sys
import numpy as np

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from longctx.analyzer import separate_failures

    m = {"disentangled": 0.0}
    try:
        res = separate_failures(np.array([0.05]), np.array([101, 102]))
        if res == "attention_failure":
            m["disentangled"] = 1.0
    except Exception:
        pass
    return m
