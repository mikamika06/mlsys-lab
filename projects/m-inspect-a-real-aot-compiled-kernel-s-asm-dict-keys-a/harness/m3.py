import os
import importlib.util

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_naive_counter": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import inspector.stages as s
    good = s.compare_num_stages

    def naive_counter(asm2, asm4):
        p2 = asm2.get("ptx", "")
        p4 = asm4.get("ptx", "")
        def c(p):
            if not p: return 0
            return len(p.splitlines())
        return {
            "size_2": len(p2), "size_4": len(p4), "size_diff": len(p4) - len(p2),
            "inst_2": c(p2), "inst_4": c(p4), "inst_diff": c(p4) - c(p2)
        }

    s.compare_num_stages = naive_counter
    try:
        out["catches_naive_counter"] = 0.0 if _survives(path) else 1.0
    finally:
        s.compare_num_stages = good

    return out
