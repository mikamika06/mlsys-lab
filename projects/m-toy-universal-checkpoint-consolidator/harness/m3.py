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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_axis": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import checkpoint.merge as m
    good = m.merge_tp_shards

    def bad_merge(shards, axis_map):
        out_ = {}
        for k in shards[0]:
            axis = axis_map.get(k)
            if axis is not None:
                out_[k] = np.concatenate([s[k] for s in shards], axis=0)
            else:
                out_[k] = shards[0][k]
        return out_

    m.merge_tp_shards = bad_merge
    import checkpoint
    checkpoint.merge_tp_shards = bad_merge

    try:
        out["catches_bad_axis"] = 0.0 if _survives(path) else 1.0
    finally:
        m.merge_tp_shards = good
        checkpoint.merge_tp_shards = good

    return out
