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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_undercounting": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on reference implementation: {e}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import kvquant.memory as mem
    orig_calc = mem.calculate_kv_cache_bytes

    def faulty_calculate(n_layers, n_kv_heads, head_dim, seq_len, quant_type="f16"):
        res = orig_calc(n_layers, n_kv_heads, head_dim, seq_len, quant_type)
        if quant_type.lower() in ("q8_0", "q4_0"):
            return res // 2
        return res

    mem.calculate_kv_cache_bytes = faulty_calculate
    import kvquant
    kvquant.memory.calculate_kv_cache_bytes = faulty_calculate

    try:
        caught = not _survives(path)
        out["catches_undercounting"] = 1.0 if caught else 0.0
    finally:
        mem.calculate_kv_cache_bytes = orig_calc
        kvquant.memory.calculate_kv_cache_bytes = orig_calc

    return out
