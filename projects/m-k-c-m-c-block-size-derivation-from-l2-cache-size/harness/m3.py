"""Milestone 3 harness check safeguard."""

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_tile": 0.0}
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

    import blocking.cache as c
    good_fn = c.derive_l2_blocking

    def broken_l2_blocking(l2_size_bytes, m_r, n_r, elem_size_bytes, alpha=0.75):
        return (m_r * 10000, 10000)

    c.derive_l2_blocking = broken_l2_blocking
    import blocking
    blocking.cache.derive_l2_blocking = broken_l2_blocking

    try:
        out["catches_invalid_tile"] = 0.0 if _survives(path) else 1.0
    finally:
        c.derive_l2_blocking = good_fn
        blocking.cache.derive_l2_blocking = good_fn

    return out
