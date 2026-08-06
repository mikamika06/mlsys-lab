import importlib.util
import os
import sys


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
        "catches_gqa_head_dim_bug": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {e}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import vllm_budget.kv as kv_mod

    orig_bytes_per_token = kv_mod.bytes_per_token

    def buggy_bytes_per_token(config, dtype):
        num_layers = config["num_hidden_layers"]
        num_heads = config["num_attention_heads"]
        head_dim = config["hidden_size"] // config["num_attention_heads"]
        elem_bytes = kv_mod.DTYPE_SIZES[dtype.lower()]
        return 2 * num_layers * num_heads * head_dim * elem_bytes

    kv_mod.bytes_per_token = buggy_bytes_per_token
    try:
        out["catches_gqa_head_dim_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        kv_mod.bytes_per_token = orig_bytes_per_token

    return out
