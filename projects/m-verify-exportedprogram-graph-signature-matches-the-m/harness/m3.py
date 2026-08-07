import importlib.util
import os
import sys

sys.path.insert(0, ".")


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_signature_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import export_verify.verifier as verifier_mod

    orig_fn = verifier_mod.verify_graph_signature

    def broken_verify_graph_signature(mod, ep, args, kwargs=None):
        valid, details = orig_fn(mod, ep, args, kwargs)
        details["param_shapes_ok"] = True
        details["buffer_shapes_ok"] = True
        valid = details["params_ok"] and details["buffers_ok"] and details["inputs_ok"]
        return valid, details

    verifier_mod.verify_graph_signature = broken_verify_graph_signature
    import export_verify

    export_verify.verifier.verify_graph_signature = broken_verify_graph_signature

    try:
        out["catches_signature_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        verifier_mod.verify_graph_signature = orig_fn
        export_verify.verifier.verify_graph_signature = orig_fn

    return out
