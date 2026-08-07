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


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_premature_zeroing": 0.0}
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

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

    import gradaccum.accumulator as accum
    orig_run = accum.run_correct_accumulation

    accum.run_correct_accumulation = accum.run_buggy_accumulation

    try:
        survived = False
        try:
            _run(path)
            survived = True
        except Exception:
            survived = False

        if not survived:
            out["catches_premature_zeroing"] = 1.0
        else:
            out["_note"] = "learner test passed even when per-micro-batch zero_grad bug was injected"
    finally:
        accum.run_correct_accumulation = orig_run

    return out
