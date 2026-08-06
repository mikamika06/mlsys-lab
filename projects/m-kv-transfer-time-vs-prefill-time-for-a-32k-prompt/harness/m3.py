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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_halved_kv_size": 0.0,
        "catches_ignored_transfer": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct reference: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import disagg.sizing as s
    import disagg.transfer as t

    orig_kv = t.kv_cache_bytes
    orig_ratio = s.compute_pd_ratio

    def halved_kv(prompt_len, num_layers, num_kv_heads, head_dim, dtype_bytes=2):
        return num_layers * num_kv_heads * head_dim * prompt_len * dtype_bytes

    def no_trans_ratio(prefill_ms, transfer_ms, decode_step_ms, gen_tokens):
        return s.compute_pd_ratio(prefill_ms, 0.0, decode_step_ms, gen_tokens)

    try:
        t.kv_cache_bytes = halved_kv
        out["catches_halved_kv_size"] = 0.0 if _survives(path) else 1.0
    finally:
        t.kv_cache_bytes = orig_kv

    try:
        s.compute_pd_ratio = no_trans_ratio
        out["catches_ignored_transfer"] = 0.0 if _survives(path) else 1.0
    finally:
        s.compute_pd_ratio = orig_ratio

    return out
