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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_bias_bug": 0.0,
        "catches_save_modules_bug": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import loraparams.compare as comp
    import loraparams.formula as form

    good_calc = form.calculate_trainable_params
    good_audit = comp.audit_recorded_run

    def broken_calc_bias(model_config, lora_config):
        res = good_calc(model_config, lora_config)
        res["total_trainable_params"] -= res["bias_params"]
        res["bias_params"] = 0
        return res

    form.calculate_trainable_params = broken_calc_bias
    try:
        out["catches_bias_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        form.calculate_trainable_params = good_calc

    def broken_calc_save(model_config, lora_config):
        res = good_calc(model_config, lora_config)
        res["total_trainable_params"] -= res["modules_to_save_params"]
        res["modules_to_save_params"] = 0
        return res

    form.calculate_trainable_params = broken_calc_save
    try:
        out["catches_save_modules_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        form.calculate_trainable_params = good_calc

    return out
