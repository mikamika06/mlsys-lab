"""Checker for Milestone 3: Regression test safeguard against invalid model pairs."""

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_pairs": 0.0}
    sys.path.insert(0, workdir)
    test_path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(test_path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(test_path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on reference code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import specdec.pair as pmod
    good_func = pmod.is_valid_draft_target_pair

    def broken_func(draft, target):
        return True

    pmod.is_valid_draft_target_pair = broken_func
    import specdec
    specdec.pair.is_valid_draft_target_pair = broken_func

    try:
        if not _survives(test_path):
            out["catches_invalid_pairs"] = 1.0
        else:
            out["_note"] = "Tests passed even when validation always returned True for invalid pairs"
    finally:
        pmod.is_valid_draft_target_pair = good_func
        specdec.pair.is_valid_draft_target_pair = good_func

    return out
