import os
import numpy as np
import time
import ref

def check(workdir):
    m = {"latency_ratio_ok": 0.0}
    mod_path = os.path.join(workdir, "ov_engine", "runtime.py")
    if not os.path.isfile(mod_path):
        return m

    import sys
    sys.path.insert(0, workdir)
    from ov_engine.runtime import run_inference

    inp = np.zeros((1, 16), dtype=np.float32)
    start_t = time.time()
    try:
        for _ in range(20):
            run_inference("dummy", inp, threads=4, latency_hint=True)
        dur = time.time() - start_t
        base = ref.get_baseline_duration()
        if dur < base * 100.0:
            m["latency_ratio_ok"] = 1.0
    except Exception:
        pass
    return m
