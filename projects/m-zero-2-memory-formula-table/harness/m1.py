import ref
import numpy as np

def check(workdir):
    from zerotwo.memory import zero2_memory_breakdown
    max_rel_err = 0.0
    for cfg in ref.TEST_CONFIGS:
        want = ref.ref_zero2_memory(**cfg)
        got = zero2_memory_breakdown(**cfg)
        for k in want:
            w_val = want[k]
            g_val = got.get(k, 0.0)
            if w_val == 0.0:
                err = abs(g_val - w_val)
            else:
                err = abs(g_val - w_val) / abs(w_val)
            if err > max_rel_err:
                max_rel_err = err
    return {"rel_err": float(max_rel_err)}
