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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_hessian_math": 0.0}
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

    import sys
    sys.path.insert(0, workdir)
    import gptqmem.model as gm

    good = gm.simulate_timeline

    def bad_simulate(in_features, out_features, calib_samples, seq_len):
        w_fp16 = in_features * out_features * 2
        w_q4 = (in_features * out_features) // 2
        acts = calib_samples * seq_len * in_features * 2
        hessian = in_features * in_features * 2

        return [
            {"phase": "load_weights", "weights": w_fp16, "hessian": 0, "activations": 0},
            {"phase": "load_activations", "weights": w_fp16, "hessian": 0, "activations": acts},
            {"phase": "compute_hessian", "weights": w_fp16, "hessian": hessian, "activations": acts},
            {"phase": "quantize", "weights": w_q4, "hessian": hessian, "activations": acts},
            {"phase": "done", "weights": w_q4, "hessian": 0, "activations": 0}
        ]

    gm.simulate_timeline = bad_simulate
    try:
        out["catches_bad_hessian_math"] = 0.0 if _survives(path) else 1.0
    finally:
        gm.simulate_timeline = good
        sys.path.pop(0)

    return out
