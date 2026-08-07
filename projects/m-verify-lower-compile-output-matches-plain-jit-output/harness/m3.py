import importlib.util
import os
import sys
import numpy as np
import ref


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_errors": 0.0}
    sys.path.insert(0, workdir)

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct reference: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import jaxinspect.verify as v
    good_verify = v.verify_compile_vs_jit

    def broken_verify(aot_outputs, jit_outputs):
        return {"max_abs_err": 0.0, "is_close": True}

    v.verify_compile_vs_jit = broken_verify
    import jaxinspect
    jaxinspect.verify.verify_compile_vs_jit = broken_verify

    try:
        if not _survives(path):
            out["catches_ignored_errors"] = 1.0
        else:
            out["_note"] = "test suite passed on broken verify implementation that ignores numerical errors"
    finally:
        v.verify_compile_vs_jit = good_verify
        jaxinspect.verify.verify_compile_vs_jit = good_verify

    return out
