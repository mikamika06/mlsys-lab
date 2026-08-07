import importlib.util
import os
import numpy as np


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
        "catches_shape_mismatch": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"the tests fail on a correct plan: {type(e).__name__}: {str(e)[:120]}"
        )
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import ovruntime.infer as infer_mod

    orig_infer = infer_mod.InferRequest.infer
    orig_start_async = infer_mod.InferRequest.start_async

    def faulty_infer(self, inputs):
        inputs = np.asarray(inputs, dtype=np.float32)
        expected = self.compiled_model.input_shape
        if inputs.shape != expected:
            inputs = np.resize(inputs, expected)
        x = inputs
        for layer in self.compiled_model.config["layers"]:
            x = np.dot(x, layer["weights"]) + layer["bias"]
            if layer.get("activation") == "relu":
                x = np.maximum(0.0, x)
        self.output = x
        self.latency_ticks = 10
        return self.output

    def faulty_start_async(self, inputs):
        inputs = np.asarray(inputs, dtype=np.float32)
        expected = self.compiled_model.input_shape
        if inputs.shape != expected:
            inputs = np.resize(inputs, expected)
        self._pending_input = inputs

    infer_mod.InferRequest.infer = faulty_infer
    infer_mod.InferRequest.start_async = faulty_start_async

    try:
        survived = _survives(path)
        out["catches_shape_mismatch"] = 0.0 if survived else 1.0
    finally:
        infer_mod.InferRequest.infer = orig_infer
        infer_mod.InferRequest.start_async = orig_start_async

    return out
