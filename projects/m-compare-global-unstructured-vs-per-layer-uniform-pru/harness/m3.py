"""Checker for Milestone 3: Regression Safeguard Test Runner."""

import importlib.util
import os


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
    """Verify test suite exists, passes on valid code, and catches injected bug."""
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_global_uniform_mixup": 0.0,
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
            f"the tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        )
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import prunecomp.pruners as p

    orig_global = p.compute_global_unstructured_mask

    def broken_global(weights, sparsity_ratio):
        masks, threshs = p.compute_per_layer_uniform_masks(weights, sparsity_ratio)
        return masks, sum(threshs.values()) / len(threshs)

    p.compute_global_unstructured_mask = broken_global
    import prunecomp

    prunecomp.pruners.compute_global_unstructured_mask = broken_global

    try:
        out["catches_global_uniform_mixup"] = 0.0 if _survives(path) else 1.0
    finally:
        p.compute_global_unstructured_mask = orig_global
        prunecomp.pruners.compute_global_unstructured_mask = orig_global

    return out
