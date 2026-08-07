import importlib.util
import os
import sys


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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_broken_registry": 0.0,
        "catches_broken_runner": 0.0,
        "faults_caught": 0.0,
    }

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on good implementation: {type(e).__name__}: {e}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import exporter.custom_ops as custom_ops
    import exporter.runtime_runner as runtime_runner

    orig_meta = custom_ops.CustomOpRegistry.meta_impl

    def broken_meta(self, x_shape, weight_shape):
        return (x_shape[0], x_shape[1], 999)

    custom_ops.CustomOpRegistry.meta_impl = broken_meta
    try:
        out["catches_broken_registry"] = 0.0 if _survives(path) else 1.0
    finally:
        custom_ops.CustomOpRegistry.meta_impl = orig_meta

    orig_run = runtime_runner.StandaloneAOTRunner.run

    def broken_run(self, inputs):
        res = orig_run(self, inputs)
        return res * 0.0

    runtime_runner.StandaloneAOTRunner.run = broken_run
    try:
        out["catches_broken_runner"] = 0.0 if _survives(path) else 1.0
    finally:
        runtime_runner.StandaloneAOTRunner.run = orig_run

    out["faults_caught"] = out["catches_broken_registry"] + out["catches_broken_runner"]
    return out
