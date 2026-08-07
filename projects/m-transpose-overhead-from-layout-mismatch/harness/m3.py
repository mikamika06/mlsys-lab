import importlib.util
import os
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_layout_mismatch": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct pipeline code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import edgeio.pipeline as p
    good_app = p.preprocess_app_side

    def broken_app_side_layout(raw_frames, mean, std):
        m = np.array(mean, dtype=np.float32).reshape(1, 1, 1, -1)
        s = np.array(std, dtype=np.float32).reshape(1, 1, 1, -1)
        return (raw_frames.astype(np.float32) / 255.0 - m) / s

    p.preprocess_app_side = broken_app_side_layout
    import edgeio
    edgeio.pipeline.preprocess_app_side = broken_app_side_layout

    try:
        out["catches_layout_mismatch"] = 0.0 if _survives(path) else 1.0
    finally:
        p.preprocess_app_side = good_app
        edgeio.pipeline.preprocess_app_side = good_app

    return out
