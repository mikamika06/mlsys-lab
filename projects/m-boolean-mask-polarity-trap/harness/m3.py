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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_inverted_mask": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        sys.path.pop(0)
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        sys.path.pop(0)
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import sdpa_trap.attention as att
    import torch.nn.functional as F

    good = att.compute_sdpa

    def broken_sdpa(q, k, v, pad_mask):
        B, H, S, D = q.shape
        mask = pad_mask.view(B, 1, 1, -1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    att.compute_sdpa = broken_sdpa
    try:
        out["catches_inverted_mask"] = 0.0 if _survives(path) else 1.0
        if out["catches_inverted_mask"] == 0.0:
            out["_note"] = "tests did not fail when compute_sdpa incorrectly passed pad_mask without inverting"
    finally:
        att.compute_sdpa = good
        sys.path.pop(0)

    return out
