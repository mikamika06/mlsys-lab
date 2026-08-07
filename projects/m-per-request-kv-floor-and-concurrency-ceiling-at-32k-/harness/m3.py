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

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_missing_tp_scaling": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests failed on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import kvcapacity.feasibility as fe
    import kvcapacity.floor as fl

    orig_cc = fe.concurrency_ceiling

    def buggy_concurrency_ceiling(gpu_memory_gb, model_config, tp_size, model_dtype, kv_dtype, seq_len, gpu_memory_utilization=0.9):
        num_kv_heads = int(model_config.get("num_key_value_heads", model_config.get("num_attention_heads", 32)))
        if tp_size <= 0 or num_kv_heads % tp_size != 0:
            return 0
        total_gpu_bytes = gpu_memory_gb * (1024**3) * gpu_memory_utilization
        weight_bytes_per_gpu = fl.model_weights_bytes(model_config, model_dtype) / tp_size
        available_kv_bytes = total_gpu_bytes - weight_bytes_per_gpu
        if available_kv_bytes <= 0:
            return 0
        kv_bytes_full = fl.per_request_kv_bytes(model_config, seq_len, kv_dtype)
        return max(0, int(available_kv_bytes // kv_bytes_full))

    fe.concurrency_ceiling = buggy_concurrency_ceiling

    try:
        failed = False
        try:
            _run(path)
        except Exception:
            failed = True
        out["catches_missing_tp_scaling"] = 1.0 if failed else 0.0
    finally:
        fe.concurrency_ceiling = orig_cc

    return out
