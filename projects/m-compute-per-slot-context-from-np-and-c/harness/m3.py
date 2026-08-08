import importlib.util
import os
import sys

import ref


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
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_broken_allocation": 0.0,
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

    import llamaslot.context as ctx_mod

    orig_compute = ctx_mod.compute_slot_context

    def broken_compute(ctx_size, n_parallel, model_max_ctx=4096):
        return (model_max_ctx if ctx_size == 0 else ctx_size) * n_parallel

    ctx_mod.compute_slot_context = broken_compute
    import llamaslot

    llamaslot.context.compute_slot_context = broken_compute

    try:
        survived = _survives(path)
        out["catches_broken_allocation"] = 0.0 if survived else 1.0
    finally:
        ctx_mod.compute_slot_context = orig_compute
        llamaslot.context.compute_slot_context = orig_compute

    return out
