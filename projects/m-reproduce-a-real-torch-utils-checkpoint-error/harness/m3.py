import importlib.util
import os
import ref


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_interval": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import chkpt.optimal as opt_mod
    good_func = opt_mod.optimal_interval

    def broken_func(n_layers, memory_limit):
        return -1

    opt_mod.optimal_interval = broken_func
    import chkpt
    if hasattr(chkpt, "optimal_interval"):
        chkpt.optimal_interval = broken_func

    try:
        out["catches_broken_interval"] = 0.0 if _survives(path) else 1.0
    finally:
        opt_mod.optimal_interval = good_func
        if hasattr(chkpt, "optimal_interval"):
            chkpt.optimal_interval = good_func
    return out
