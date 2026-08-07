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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_alignment_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        res = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on good code: {e}"
        return out

    if res is None:
        out["_note"] = "No test_ functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import rectatt.probe as probe_mod
    orig_compute_offset = probe_mod.compute_offset

    def buggy_compute_offset(n_q: int, n_kv: int, alignment: str) -> int:
        return 0

    probe_mod.compute_offset = buggy_compute_offset

    import rectatt.attention as attn_mod
    orig_sdpa = attn_mod.sdpa_rectangular_causal

    def buggy_sdpa(q, k, v, alignment="bottom_right"):
        return orig_sdpa(q, k, v, alignment="top_left")

    attn_mod.sdpa_rectangular_causal = buggy_sdpa

    try:
        survived = _survives(path)
        out["catches_alignment_bug"] = 0.0 if survived else 1.0
    finally:
        probe_mod.compute_offset = orig_compute_offset
        attn_mod.sdpa_rectangular_causal = orig_sdpa

    return out
