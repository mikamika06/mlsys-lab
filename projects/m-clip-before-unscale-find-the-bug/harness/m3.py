import importlib.util
import os
import torch


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_order": 0.0}
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

    import scalerlab.clip as sc
    good_fn = sc.perform_optimizer_step

    def bad_perform_optimizer_step(model, optimizer, scaler, max_norm):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

    sc.perform_optimizer_step = bad_perform_optimizer_step
    try:
        out["catches_bad_order"] = 0.0 if _survives(path) else 1.0
    finally:
        sc.perform_optimizer_step = good_fn

    return out
