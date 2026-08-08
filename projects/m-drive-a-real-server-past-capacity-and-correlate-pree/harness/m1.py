import ref
import numpy as np

def check(workdir):
    from serverload.load import simulate_load
    out = {"loads_matched": 0.0}
    ok = 0
    for sc in ref.SCENARIOS:
        want = ref.simulate_load(sc)
        got = simulate_load(sc)
        if np.allclose(want["preemptions"], got["preemptions"], rtol=1e-5) and np.allclose(want["latencies"], got["latencies"], rtol=1e-5):
            ok += 1
    out["loads_matched"] = float(ok)
    return out
