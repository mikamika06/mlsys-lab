import ref
import numpy as np


def check(workdir):
    from syncbug.bench import reduce_time_ratio
    errs = []
    for args in ref.TEST_CASES_BENCH:
        want = ref.reduce_time_ratio(*args)
        try:
            got = reduce_time_ratio(*args)
        except Exception as e:
            return {"rel_err": 1.0, "_note": f"Exception: {type(e).__name__}: {e}"}
        rel = abs(got - want) / (abs(want) + 1e-8)
        errs.append(rel)
    mean_err = float(np.mean(errs)) if errs else 1.0
    return {"rel_err": mean_err}
