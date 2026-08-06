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
    for k in list(sys.modules.keys()):
        if k == "aot_compare" or k.startswith("aot_compare."):
            del sys.modules[k]

    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_fault": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import aot_compare.stablehlo_diff as sd
    good_diff = sd.diff_stablehlo_ops

    def broken_diff(jit_fn, args, flags_a=None, flags_b=None):
        ops_a = sd.get_stablehlo_ops(jit_fn, args, flags_a)
        ops_b = sd.get_stablehlo_ops(jit_fn, args, flags_b)
        all_ops = sorted(set(ops_a.keys()) | set(ops_b.keys()))
        return {op: ops_a.get(op, 0) - ops_b.get(op, 0) for op in all_ops}

    sd.diff_stablehlo_ops = broken_diff
    import aot_compare
    aot_compare.diff_stablehlo_ops = broken_diff

    try:
        out["catches_fault"] = 0.0 if _survives(path) else 1.0
    finally:
        sd.diff_stablehlo_ops = good_diff
        aot_compare.diff_stablehlo_ops = good_diff

    return out
