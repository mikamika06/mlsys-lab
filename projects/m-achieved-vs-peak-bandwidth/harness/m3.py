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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_underaccounted_bytes": 0.0}
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
        out["_note"] = f"the tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import bandwidth.tracker as tracker_mod

    good_fn = tracker_mod.compute_bytes_transferred

    def broken_compute_bytes_transferred(config):
        b = config["batch_size"]
        h = config["num_heads"]
        n = config["seq_len"]
        d = config["head_dim"]
        p = config["element_bytes"]
        naive = float(b * h * p * (4 * n * d + 2 * n * n))
        tiled = float(b * h * n * d * p * (2 + 2 * 1))
        return {"naive_bytes": naive, "tiled_bytes": tiled}

    tracker_mod.compute_bytes_transferred = broken_compute_bytes_transferred

    import bandwidth

    bandwidth.tracker.compute_bytes_transferred = broken_compute_bytes_transferred

    try:
        catches = not _survives(path)
        out["catches_underaccounted_bytes"] = 1.0 if catches else 0.0
    finally:
        tracker_mod.compute_bytes_transferred = good_fn
        bandwidth.tracker.compute_bytes_transferred = good_fn

    return out
