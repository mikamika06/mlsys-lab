import os
import numpy as np

def check(workdir):
    m = {"profile_ok": 0.0}
    mod_path = os.path.join(workdir, "ov_engine", "profiler.py")
    if not os.path.isfile(mod_path):
        return m

    import sys
    sys.path.insert(0, workdir)
    from ov_engine.profiler import profile_model

    dummy = np.zeros((1, 16), dtype=np.float32)
    try:
        profile = profile_model("dummy", dummy)
        if isinstance(profile, dict) and len(profile) > 0:
            m["profile_ok"] = 1.0
    except Exception:
        pass
    return m
