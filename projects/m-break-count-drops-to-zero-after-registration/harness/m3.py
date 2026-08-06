import importlib.util
import os
import torch


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unregistered_op": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import customop.ops as ops_mod
    orig_register = ops_mod.register_custom_op

    def broken_register():
        def raw_op(x, alpha):
            torch._dynamo.graph_break()
            return ops_mod.raw_smooth_relumix(x, alpha)
        return raw_op

    ops_mod.register_custom_op = broken_register
    try:
        survived = _survives(path)
        out["catches_unregistered_op"] = 0.0 if survived else 1.0
    finally:
        ops_mod.register_custom_op = orig_register

    return out
