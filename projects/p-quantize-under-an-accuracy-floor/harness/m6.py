import importlib.util
import os


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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_broken_sensitivity": 0.0,
        "catches_broken_mixed_prec": 0.0,
        "faults_caught": 0.0,
    }

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import quant.sensitivity as sens_mod

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"the tests fail on a correct implementation: {type(e).__name__}: {e}"
        )
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    orig_sens = sens_mod.compute_layer_sensitivity
    sens_mod.compute_layer_sensitivity = lambda model, evaluator, calib: {
        k: 0.0 for k in model.layers
    }
    try:
        out["catches_broken_sensitivity"] = 0.0 if _survives(path) else 1.0
    finally:
        sens_mod.compute_layer_sensitivity = orig_sens

    orig_config = sens_mod.select_mixed_precision_config
    sens_mod.select_mixed_precision_config = (
        lambda sens, target_bits=5.5: {k: 2 for k in sens}
    )
    try:
        out["catches_broken_mixed_prec"] = 0.0 if _survives(path) else 1.0
    finally:
        sens_mod.select_mixed_precision_config = orig_config

    out["faults_caught"] = (
        out["catches_broken_sensitivity"] + out["catches_broken_mixed_prec"]
    )
    return out
