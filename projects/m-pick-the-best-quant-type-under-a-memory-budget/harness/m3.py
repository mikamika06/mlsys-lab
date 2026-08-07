import importlib.util
import os
import sys


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_fallback": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    if workdir not in sys.path:
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

    import quantplan.backend as backend_mod
    import quantplan.picker as picker_mod

    orig_backend_fb = backend_mod.will_fallback_to_cpu
    orig_picker_fb = getattr(picker_mod, "will_fallback_to_cpu", None)

    def broken_fallback(quant_type, backend_config):
        return False

    backend_mod.will_fallback_to_cpu = broken_fallback
    if hasattr(picker_mod, "will_fallback_to_cpu"):
        picker_mod.will_fallback_to_cpu = broken_fallback

    try:
        failed = not _survives(path)
        out["catches_ignored_fallback"] = 1.0 if failed else 0.0
        if not failed:
            out["_note"] = "tests passed even when IQ CPU fallback was completely ignored"
    finally:
        backend_mod.will_fallback_to_cpu = orig_backend_fb
        if orig_picker_fb is not None:
            picker_mod.will_fallback_to_cpu = orig_picker_fb

    return out
