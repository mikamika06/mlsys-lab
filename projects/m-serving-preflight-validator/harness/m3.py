import importlib.util
import os
import sys

sys.path.insert(0, ".")


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_oom_approvals": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on valid reference: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import preflight.validator as v

    good_fit = v.validate_fit

    def broken_validate_fit(model_config, quant_config, gpu_specs):
        res = good_fit(model_config, quant_config, gpu_specs)
        res["fits"] = True
        return res

    v.validate_fit = broken_validate_fit
    import preflight

    preflight.validator.validate_fit = broken_validate_fit

    try:
        out["catches_oom_approvals"] = 0.0 if _survives(path) else 1.0
    finally:
        v.validate_fit = good_fit
        preflight.validator.validate_fit = good_fit

    return out
