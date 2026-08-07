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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_memory_regression": 0.0,
        "catches_batch_drop": 0.0,
        "faults_caught": 0.0
    }
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import qlora_fix.memory as mem
    import qlora_fix.optimizer as opt

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_account = mem.account_memory
    mem.account_memory = lambda w, a, o: {"total_mb": 25000, "valid": True}
    try:
        out["catches_memory_regression"] = 0.0 if _survives(path) else 1.0
    finally:
        mem.account_memory = good_account

    good_step = opt.run_training_step
    opt.run_training_step = lambda model, batch, accum: {"completed": True, "effective_batch": 2}
    try:
        out["catches_batch_drop"] = 0.0 if _survives(path) else 1.0
    finally:
        opt.run_training_step = good_step

    out["faults_caught"] = out["catches_memory_regression"] + out["catches_batch_drop"]
    return out
