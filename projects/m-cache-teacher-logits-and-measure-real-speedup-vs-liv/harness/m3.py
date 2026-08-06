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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_live_fallback": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct plan: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import distillcache.engine as eng
    good_epoch = eng.run_distillation_epoch

    def broken_epoch(student_model, dataset, cache=None, teacher_model=None, mode="offline"):
        if mode == "offline" and teacher_model is not None:
            for sample in dataset:
                teacher_model(sample["input"])
        return good_epoch(student_model, dataset, cache=cache, teacher_model=teacher_model, mode=mode)

    eng.run_distillation_epoch = broken_epoch
    import distillcache
    distillcache.engine.run_distillation_epoch = broken_epoch
    try:
        out["catches_live_fallback"] = 0.0 if _survives(path) else 1.0
    finally:
        eng.run_distillation_epoch = good_epoch
        distillcache.engine.run_distillation_epoch = good_epoch

    return out
