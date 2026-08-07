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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_leaky_routing": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on valid reference implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import loraserving.routing as r

    good = r.apply_per_row_lora

    def leaky_routing(x, adapter_ids, lora_a, lora_b, scaling):
        out_tensor = good(x, adapter_ids, lora_a, lora_b, scaling)
        if len(out_tensor) > 1:
            out_tensor[0] = out_tensor[1]
        return out_tensor

    r.apply_per_row_lora = leaky_routing
    import loraserving
    loraserving.routing.apply_per_row_lora = leaky_routing

    try:
        out["catches_leaky_routing"] = 0.0 if _survives(path) else 1.0
    finally:
        r.apply_per_row_lora = good
        loraserving.routing.apply_per_row_lora = good

    return out
