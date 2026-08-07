import importlib.util
import os
import sys
import numpy as np


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_naive_merge": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on valid implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import onlinesoftmax.merge as target_mod

    orig_merge = target_mod.merge_online_softmax

    def faulty_naive_merge(m_a, l_a, o_a, m_b, l_b, o_b):
        m_new = np.maximum(m_a, m_b)
        l_new = l_a + l_b
        o_new = (o_a + o_b) / 2.0
        return m_new, l_new, o_new

    target_mod.merge_online_softmax = faulty_naive_merge
    try:
        out["catches_naive_merge"] = 0.0 if _survives(path) else 1.0
    finally:
        target_mod.merge_online_softmax = orig_merge

    return out
