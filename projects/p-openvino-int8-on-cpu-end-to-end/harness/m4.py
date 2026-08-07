import os
import numpy as np

def check(workdir):
    m = {"threads_configured": 0.0}
    mod_path = os.path.join(workdir, "ov_engine", "runtime.py")
    if not os.path.isfile(mod_path):
        return m

    import sys
    sys.path.insert(0, workdir)
    from ov_engine.runtime import run_inference

    inp = np.zeros((1, 16), dtype=np.float32)
    try:
        out = run_inference("dummy", inp, threads=4, latency_hint=True)
        if out is not None:
            m["threads_configured"] = 1.0
    except Exception:
        pass
    return m
