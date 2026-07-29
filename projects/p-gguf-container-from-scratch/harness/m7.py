import importlib.util
import os

import w


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0,
           "catches_truncated": 0.0, "catches_lying_count": 0.0}
    if w.gguf_or_none() is None:
        return w.needs_gguf(out)
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct writer: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import gguf_lab.writer as gw
    good = gw.write

    def truncating(path_, *a, **kw):
        good(path_, *a, **kw)
        with open(path_, "r+b") as f:
            f.truncate(max(0, os.path.getsize(path_) - 24))
        return path_

    gw.write = truncating
    try:
        out["catches_truncated"] = 0.0 if _survives(path) else 1.0
    finally:
        gw.write = good

    def lying(path_, arch, metadata, tensors, alignment=32):
        good(path_, arch, metadata, tensors, alignment=alignment)
        with open(path_, "r+b") as f:
            f.seek(8)
            f.write((len(tensors) + 3).to_bytes(8, "little"))
        return path_

    gw.write = lying
    try:
        out["catches_lying_count"] = 0.0 if _survives(path) else 1.0
    finally:
        gw.write = good
    return out
