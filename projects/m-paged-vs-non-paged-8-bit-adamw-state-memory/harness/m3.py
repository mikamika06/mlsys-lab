import importlib.util
import os
import sys
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unpaged_leak": 0.0, "catches_unscaled_merge": 0.0}
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
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import qlora_mem.adamw as adamw_mod
    import qlora_mem.lora_merge as lora_mod

    good_adamw = adamw_mod.compute_adamw_state_bytes
    good_merge = lora_mod.merge_lora_into_base

    def bad_adamw(num_params, block_size=256, paged=False, max_layer_params=0):
        import math
        blocks = math.ceil(num_params / block_size)
        return num_params * 2 + blocks * 8

    def bad_merge(qweights, scales, lora_A, lora_B, alpha, block_size=64):
        unscaled_dequant = lora_mod.CODEBOOK_4BIT[qweights].astype(np.float32)
        r = lora_A.shape[0]
        scaling = alpha / float(r)
        delta = (lora_B @ lora_A) * scaling
        return unscaled_dequant + delta

    adamw_mod.compute_adamw_state_bytes = bad_adamw
    try:
        out["catches_unpaged_leak"] = 0.0 if _survives(path) else 1.0
    finally:
        adamw_mod.compute_adamw_state_bytes = good_adamw

    lora_mod.merge_lora_into_base = bad_merge
    try:
        out["catches_unscaled_merge"] = 0.0 if _survives(path) else 1.0
    finally:
        lora_mod.merge_lora_into_base = good_merge

    return out
