import importlib.util
import os


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_fallback": 0.0}
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

    import kvquant
    import kvquant.capacity as cap

    good_max_context = cap.max_context_length

    def broken_max_context(vram_budget_bytes, base_model_bytes, model_cfg, k_type="q8_0", v_type="q8_0"):
        avail = vram_budget_bytes - base_model_bytes
        if avail <= 0:
            return 0
        tb = getattr(cap, "TYPE_BYTES", {"f32": 4.0, "f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625, "q4_1": 0.625})
        elements_per_token = model_cfg["n_layers"] * model_cfg["n_kv_heads"] * model_cfg["head_dim"]
        unit_bytes = elements_per_token * (tb[k_type] + tb[v_type])
        return int(avail // unit_bytes)

    cap.max_context_length = broken_max_context
    if hasattr(kvquant, "max_context_length"):
        kvquant.max_context_length = broken_max_context

    try:
        out["catches_ignored_fallback"] = 0.0 if _survives(path) else 1.0
    finally:
        cap.max_context_length = good_max_context
        if hasattr(kvquant, "max_context_length"):
            kvquant.max_context_length = good_max_context

    return out
