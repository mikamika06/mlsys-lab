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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_freed_compute": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "test_regression.py missing"
        return out

    sys.path.insert(0, workdir)
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {e}"
        sys.path.pop(0)
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        sys.path.pop(0)
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import zero3.schedule
    good_build = zero3.schedule.build_schedule

    def broken_build(num_layers, prefetch):
        s = good_build(num_layers, prefetch)
        try:
            # Injecting a fault by swapping compute and free, dropping active memory
            c_idx = s.index(("compute_fw", 1))
            f_idx = s.index(("free_fw", 1))
            s[c_idx], s[f_idx] = s[f_idx], s[c_idx]
        except ValueError:
            pass
        return s

    zero3.schedule.build_schedule = broken_build
    try:
        out["catches_freed_compute"] = 0.0 if _survives(path) else 1.0
    finally:
        zero3.schedule.build_schedule = good_build
        sys.path.pop(0)

    return out
