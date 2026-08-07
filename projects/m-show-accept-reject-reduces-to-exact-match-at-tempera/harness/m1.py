import ref
import numpy as np

def check(workdir):
    from specsampling.core import verify_zero_temp_reduction
    out = {"reduction_matched": 0.0}
    np.random.seed(123)
    ok = 0
    trials = 10
    for _ in range(trials):
        t_logits = np.random.randn(20)
        d_logits = np.random.randn(20)
        want = ref.verify_zero_temp_reduction(t_logits, d_logits)
        try:
            got = verify_zero_temp_reduction(t_logits, d_logits)
            if bool(got) == bool(want):
                ok += 1
        except Exception:
            pass
    out["reduction_matched"] = 1.0 if ok == trials else 0.0
    return out
