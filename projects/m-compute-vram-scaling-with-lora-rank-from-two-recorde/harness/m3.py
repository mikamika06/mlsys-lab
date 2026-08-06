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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_zero_intercept_fault": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Learner tests failed on valid reference code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import lorascaling.predictor as pred_mod
    good_predict = pred_mod.predict_rank_requirements

    def zero_intercept_predict(params, target_rank):
        tr = float(target_rank)
        base_v = float(params["vram_base"])
        slope_v = float(params["vram_slope"])
        eff_rate_v = (base_v + slope_v * 8.0) / 8.0

        base_f = float(params["flops_base"])
        slope_f = float(params["flops_slope"])
        eff_rate_f = (base_f + slope_f * 8.0) / 8.0

        return {
            "predicted_vram_bytes": eff_rate_v * tr,
            "predicted_step_flops": eff_rate_f * tr,
        }

    pred_mod.predict_rank_requirements = zero_intercept_predict
    import lorascaling
    if hasattr(lorascaling, "predict_rank_requirements"):
        lorascaling.predict_rank_requirements = zero_intercept_predict

    try:
        out["catches_zero_intercept_fault"] = 0.0 if _survives(path) else 1.0
    finally:
        pred_mod.predict_rank_requirements = good_predict
        if hasattr(lorascaling, "predict_rank_requirements"):
            lorascaling.predict_rank_requirements = good_predict

    return out
