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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_memory_leak": 0.0, "catches_bad_template": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import mlx_serve.memory as mem
    import mlx_serve.server as srv

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail: {type(e).__name__}: {e}"
        return out

    if first is None:
        out["_note"] = "no test functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_check = mem.check_stability
    mem.check_stability = lambda allocs: {"memory_stable_ok": 0.0}
    try:
        out["catches_memory_leak"] = 0.0 if _survives(path) else 1.0
    finally:
        mem.check_stability = good_check

    good_fmt = srv.format_chat
    srv.format_chat = lambda msgs: "bad_format"
    try:
        out["catches_bad_template"] = 0.0 if _survives(path) else 1.0
    finally:
        srv.format_chat = good_fmt

    out["faults_caught"] = out["catches_memory_leak"] + out["catches_bad_template"]
    return out
