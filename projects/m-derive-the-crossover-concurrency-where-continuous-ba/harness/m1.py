import ref
import numpy as np

def check(workdir):
    from crossover.derive import derive_crossover
    cases = ref.get_test_cases()
    errs = []
    for i, c in enumerate(cases):
        want = ref.compute_crossover(c)
        got = derive_crossover(c)
        if want is None or got is None:
            errs.append(1.0)
        else:
            rel = abs(got - want) / max(1.0, abs(want))
            errs.append(rel)
    mean_err = float(np.mean(errs)) if errs else 1.0
    return {"rel_err": mean_err}
