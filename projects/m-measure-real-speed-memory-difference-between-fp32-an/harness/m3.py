import importlib.util
import os


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_nesting": 0.0}
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

    import autoperf.nested as n_mod
    orig_fn = n_mod.run_with_nested_disable

    def broken_nested(model, x):
        device_type = "cuda" if importlib.util.find_spec("torch") else "cpu"
        import torch
        results = []
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            results.append(True)
            results.append(True)  # broken: did not disable inside
            out = model(x)
            results.append(True)
        return results, out

    n_mod.run_with_nested_disable = broken_nested
    try:
        out["catches_broken_nesting"] = 0.0 if _survives(path) else 1.0
    finally:
        n_mod.run_with_nested_disable = orig_fn
    return out
