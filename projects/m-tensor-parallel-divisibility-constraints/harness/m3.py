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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_excessive_padding": 0.0}
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
        sys.path.pop(0)
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        sys.path.pop(0)
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import tp_marlin.analyze as a
    good_pad = a.pad_for_marlin

    def excessive_pad(layers, tp_size):
        res = good_pad(layers, tp_size)
        for layer in res:
            layer["in_features"] += (128 * tp_size if layer["style"] == "row" else 128)
            layer["out_features"] += (256 if layer["style"] == "row" else 256 * tp_size)
        return res

    a.pad_for_marlin = excessive_pad
    try:
        out["catches_excessive_padding"] = 0.0 if _survives(path) else 1.0
    finally:
        a.pad_for_marlin = good_pad
        sys.path.pop(0)

    if out["catches_excessive_padding"] == 0.0:
        out["_note"] = "tests did not catch excessive padding; invariant is not minimal padding"

    return out
