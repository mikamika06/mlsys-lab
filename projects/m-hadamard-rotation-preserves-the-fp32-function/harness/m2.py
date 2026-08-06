import ref
import numpy as np

def check(workdir):
    from hadamard.stats import get_outlier_stats

    x = ref.TEST_X
    got = get_outlier_stats(x, ref.H)
    want = ref.get_outlier_stats(x)

    match = 1.0
    for k in want:
        if abs(got.get(k, 0) - want[k]) > 1e-4:
            match = 0.0
            break

    out = {"stats_matched": float(match)}
    return out
