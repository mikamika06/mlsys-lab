import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("test_regression", path)
    mod = importlib.util.module_from_spec(spec)

    workdir = os.path.dirname(os.path.dirname(path))
    sys_path_added = False
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
        sys_path_added = True

    try:
        spec.loader.exec_module(mod)
        fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
        if not fns:
            return None
        for fn in fns:
            fn()
        return True
    finally:
        if sys_path_added:
            sys.path.remove(workdir)


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_lossy_decoding": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "test_regression.py missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct simulation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    sys.path.insert(0, workdir)
    import prompt_lookup.simulate as sim
    good = sim.simulate

    def bad_simulate(*args, **kwargs):
        res = good(*args, **kwargs)
        if len(res["generated"]) > 0:
            res["generated"][-1] = -9999
        return res

    sim.simulate = bad_simulate
    try:
        out["catches_lossy_decoding"] = 0.0 if _survives(path) else 1.0
    finally:
        sim.simulate = good
        sys.path.pop(0)

    return out
