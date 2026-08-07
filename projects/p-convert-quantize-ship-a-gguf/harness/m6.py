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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on valid code: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import gguf_pipeline.quantizer as quantizer
    import gguf_pipeline.evaluator as evaluator

    good_quant = quantizer.quantize_q8_0
    quantizer.quantize_q8_0 = lambda tensor: {"qdata": np.zeros_like(tensor, dtype=np.int8), "scales": np.ones((tensor.shape[0], 1), dtype=np.float32)}

    fault1 = 0.0 if _survives(path) else 1.0
    quantizer.quantize_q8_0 = good_quant

    good_kl = evaluator.compute_kl_divergence
    evaluator.compute_kl_divergence = lambda p, q: -1.0

    fault2 = 0.0 if _survives(path) else 1.0
    evaluator.compute_kl_divergence = good_kl

    out["faults_caught"] = fault1 + fault2
    return out
