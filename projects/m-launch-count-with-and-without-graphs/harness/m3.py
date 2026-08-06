import importlib.util
import os
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_dynamic_recapture_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests failed on reference: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import launchgraph.harness as harness_mod

    old_class = harness_mod.StaticBufferHarness

    class BrokenReallocatingHarness:
        def __init__(self, max_shape, dtype=np.float32):
            self.max_shape = tuple(max_shape)
            self.dtype = dtype
            self.buffer = None

        def update_input(self, tensor):
            tensor = np.asarray(tensor, dtype=self.dtype)
            self.buffer = np.copy(tensor)
            return self.buffer

        def run(self, graph_runner):
            return graph_runner(self.buffer)

    harness_mod.StaticBufferHarness = BrokenReallocatingHarness
    try:
        out["catches_dynamic_recapture_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        harness_mod.StaticBufferHarness = old_class

    return out
