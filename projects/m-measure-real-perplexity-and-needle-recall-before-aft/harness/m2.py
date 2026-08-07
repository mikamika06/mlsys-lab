import ref
import numpy as np

def check(workdir):
    from longctx.needle import measure_recall_at_k
    out = {"recall_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        scores = np.random.randn(10, 100)
        needles = np.array([12, 45, 23, 78, 9, 33, 44, 55, 66, 77])
        k = 10
        want = ref.measure_recall_at_k(scores, needles, k)
        try:
            got = measure_recall_at_k(scores, needles, k)
            if np.isclose(got, want, rtol=1e-5, atol=1e-5):
                ok += 1
        except Exception:
            pass
    out["recall_matched"] = float(ok)
    return out
