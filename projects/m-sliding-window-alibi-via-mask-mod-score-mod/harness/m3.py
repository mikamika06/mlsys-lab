import importlib.util
import os
import numpy as np

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True

def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_causality": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    try:
        import flex.block_mask as bm
        good = bm.compute_block_mask_indices
    except ImportError:
        return out

    def broken_compute(seq_len, window, block_size):
        s, n, idxs = good(seq_len, window, block_size)

        new_max = idxs.shape[1] + 1
        new_idxs = np.zeros((idxs.shape[0], new_max), dtype=np.int32)
        new_idxs[:, :idxs.shape[1]] = idxs

        if len(n) > 0 and seq_len > block_size:
            new_idxs[0, n[0]] = 1
            n[0] += 1

        return s, n, new_idxs

    bm.compute_block_mask_indices = broken_compute
    try:
        out["catches_broken_causality"] = 0.0 if _survives(path) else 1.0
    finally:
        bm.compute_block_mask_indices = good

    return out
