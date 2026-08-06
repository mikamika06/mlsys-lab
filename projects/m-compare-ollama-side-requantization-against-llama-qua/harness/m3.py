import importlib.util
import os
import sys


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_quant_drift": 0.0}
    sys.path.insert(0, workdir)
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import runner.quant_compare as qc
    orig_fn = qc.ollama_requantize_mock

    def broken_ollama_requantize(tensor_data, quant_type="Q4_0"):
        res = orig_fn(tensor_data, quant_type=quant_type)
        return res * 5.0 + 10.0

    qc.ollama_requantize_mock = broken_ollama_requantize
    try:
        if not _survives(path):
            out["catches_quant_drift"] = 1.0
        else:
            out["_note"] = "Learner's test suite passed despite artificial quantization scale corruption"
    finally:
        qc.ollama_requantize_mock = orig_fn

    return out
