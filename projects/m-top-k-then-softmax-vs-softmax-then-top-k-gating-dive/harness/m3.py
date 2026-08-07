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
        "catches_broken_dispatch": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on valid code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import moegating.dispatch as d

    good_dispatch = d.build_mixtral_dispatch_tensor

    def broken_dispatch(selected_experts, num_experts):
        num_tokens, top_k = selected_experts.shape
        dispatch = np.zeros((num_experts, num_tokens, top_k), dtype=np.int32)
        for t in range(num_tokens):
            for k_idx in range(top_k):
                exp_id = selected_experts[t, k_idx]
                dispatch[exp_id, t, 0] = 1
        return dispatch

    d.build_mixtral_dispatch_tensor = broken_dispatch
    import moegating

    moegating.dispatch.build_mixtral_dispatch_tensor = broken_dispatch

    try:
        survived = _survives(path)
        out["catches_broken_dispatch"] = 0.0 if survived else 1.0
    finally:
        d.build_mixtral_dispatch_tensor = good_dispatch
        moegating.dispatch.build_mixtral_dispatch_tensor = good_dispatch

    return out
