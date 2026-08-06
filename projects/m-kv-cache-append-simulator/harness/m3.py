import importlib.util
import os
import sys

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_off_by_one": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import simulator.core as core
    good = core.append_tokens

    def off_by_one_append(cache_seqlens, block_tables, block_size):
        res = []
        for seqlen, table in zip(cache_seqlens, block_tables):
            err_seqlen = max(0, seqlen - 1)
            block_idx = err_seqlen // block_size
            block_offset = err_seqlen % block_size
            res.append((table[block_idx], block_offset))
        return res

    core.append_tokens = off_by_one_append
    try:
        survived = _survives(path)
        out["catches_off_by_one"] = 0.0 if survived else 1.0
        if survived:
            out["_note"] = "test suite passed even when append_tokens wrote to seqlen - 1"
    finally:
        core.append_tokens = good
    return out
