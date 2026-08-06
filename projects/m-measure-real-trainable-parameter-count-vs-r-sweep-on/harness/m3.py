import importlib.util
import os
import sys
import numpy as np
import ref


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
    sys.path.insert(0, workdir)
    from lorameasure.stochasticity import measure_dropout_stochasticity

    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_missing_rank_scaling": 0.0,
        "stochasticity_matched": 0.0
    }

    x = np.ones((4, 32), dtype=np.float64)
    w_a = np.ones((8, 32), dtype=np.float64)
    w_b = np.ones((32, 8), dtype=np.float64)

    res_det = measure_dropout_stochasticity(x, w_a, w_b, lora_alpha=16, lora_dropout=0.0, num_samples=10, seed=123)
    res_stoch = measure_dropout_stochasticity(x, w_a, w_b, lora_alpha=16, lora_dropout=0.2, num_samples=10, seed=123)

    ref_det = ref.measure_dropout_stochasticity(x, w_a, w_b, lora_alpha=16, lora_dropout=0.0, num_samples=10, seed=123)
    ref_stoch = ref.measure_dropout_stochasticity(x, w_a, w_b, lora_alpha=16, lora_dropout=0.2, num_samples=10, seed=123)

    if (res_det["is_stochastic"] == ref_det["is_stochastic"] and
            res_stoch["is_stochastic"] == ref_stoch["is_stochastic"] and
            abs(res_stoch["mean_variance"] - ref_stoch["mean_variance"]) < 1e-5):
        out["stochasticity_matched"] = 1.0

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"learner tests failed on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import lorameasure.params as p
    good_count = p.count_trainable_params

    def broken_count(model_structure, target_modules, r, use_rslora=False):
        return good_count(model_structure, target_modules, r=1, use_rslora=use_rslora)

    p.count_trainable_params = broken_count
    import lorameasure
    lorameasure.params.count_trainable_params = broken_count

    try:
        out["catches_missing_rank_scaling"] = 0.0 if _survives(path) else 1.0
    finally:
        p.count_trainable_params = good_count
        lorameasure.params.count_trainable_params = good_count

    return out
