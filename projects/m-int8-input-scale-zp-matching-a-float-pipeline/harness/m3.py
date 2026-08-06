import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_layout_mismatch": 0.0,
    }
    sys.path.insert(0, workdir)
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out[
            "_note"
        ] = f"tests fail on good pipeline: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import edgepipe.layout as l

    good_layout = l.diagnose_and_fix_layout

    def broken_layout(img, src_format, dst_format, src_order, dst_order):
        return img

    l.diagnose_and_fix_layout = broken_layout
    import edgepipe.export as e

    e.diagnose_and_fix_layout = broken_layout

    try:
        out["catches_layout_mismatch"] = 0.0 if _survives(path) else 1.0
    finally:
        l.diagnose_and_fix_layout = good_layout
        e.diagnose_and_fix_layout = good_layout

    return out
