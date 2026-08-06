import importlib.util
import os


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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_unconstrained_chooser": 0.0,
    }

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"learner tests failed on correct reference: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import layout.chooser as lc

    orig_select = lc.select_layout

    def broken_select(config, vram_gb, latency_table):
        candidate_layouts = [
            (1, 1, 8), (2, 1, 4), (4, 1, 2), (8, 1, 1),
            (1, 2, 4), (2, 2, 2), (4, 2, 1),
            (1, 4, 2), (2, 4, 1), (1, 8, 1),
        ]
        best_idx = -1
        best_lat = float("inf")
        for idx, key in enumerate(candidate_layouts):
            lat = latency_table.get(key, float("inf"))
            if lat < best_lat:
                best_lat = lat
                best_idx = idx
        return best_idx

    lc.select_layout = broken_select
    try:
        out["catches_unconstrained_chooser"] = 0.0 if _survives(path) else 1.0
    finally:
        lc.select_layout = orig_select

    return out
