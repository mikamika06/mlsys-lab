import importlib.util
import os
import sys
import numpy as np


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_static_alpha": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Learner test fails on good code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import smoothquant.autotune as sq_auto
    good_sweep = sq_auto.sweep_alpha_per_layer

    def broken_static_sweep(layer_activations, layer_weights, alpha_candidates):
        res = {}
        for name, X in layer_activations.items():
            W = layer_weights[name]
            ref_out = X @ W.T
            s = sq_auto.compute_migration_scales(np.max(np.abs(X), axis=0), np.max(np.abs(W), axis=0), 0.0)
            X_s, W_s = sq_auto.apply_smoothquant(X, W, s)
            X_q = sq_auto.quantize_int8(X_s, axis=None)
            W_q = sq_auto.quantize_int8(W_s, axis=1)
            out_q = X_q @ W_q.T
            mse = float(np.mean((ref_out - out_q) ** 2))
            res[name] = {"alpha": 0.0, "mse": mse, "scales": s}
        return res

    sq_auto.sweep_alpha_per_layer = broken_static_sweep

    try:
        out["catches_static_alpha"] = 0.0 if _survives(path) else 1.0
    finally:
        sq_auto.sweep_alpha_per_layer = good_sweep

    return out
