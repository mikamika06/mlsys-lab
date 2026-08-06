import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_test", path)
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


def check(workdir):
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_eager_expansion": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        res = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on correct code: {e}"
        return out

    if res is None:
        out["_note"] = "No test_ functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import gqa_opt.memory as mem

    orig_func = mem.calculate_kv_cache_bytes

    def faulty_calculate_kv_cache_bytes(
        config, batch_size, seq_len, dtype_bytes=2
    ):
        res = orig_func(config, batch_size, seq_len, dtype_bytes)
        res["native_bytes"] = res["mha_bytes"]
        res["bytes_saved"] = 0
        return res

    mem.calculate_kv_cache_bytes = faulty_calculate_kv_cache_bytes

    try:
        spec = importlib.util.spec_from_file_location(
            "learner_test_faulty", path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fns = [
            getattr(mod, n)
            for n in dir(mod)
            if n.startswith("test_") and callable(getattr(mod, n))
        ]
        failed = False
        for fn in fns:
            try:
                fn()
            except Exception:
                failed = True
                break
        out["catches_eager_expansion"] = 1.0 if failed else 0.0
    except Exception:
        out["catches_eager_expansion"] = 1.0
    finally:
        mem.calculate_kv_cache_bytes = orig_func

    return out
