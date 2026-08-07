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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_fallbacks": 0.0}
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

    import compiler_diag.storm as storm_mod
    original_fn = storm_mod.simulate_recompile_storm

    def broken_simulate_recompile_storm(shapes, cache_size_limit):
        compiled_shapes = set()
        history = []
        for idx, shape in enumerate(shapes):
            status = "hit" if shape in compiled_shapes else "recompile"
            compiled_shapes.add(shape)
            history.append({"step": idx, "shape": shape, "status": status})
        return {
            "history": history,
            "fallback_step": None,
            "total_recompiles": len(compiled_shapes),
            "total_fallbacks": 0
        }

    storm_mod.simulate_recompile_storm = broken_simulate_recompile_storm
    import compiler_diag
    compiler_diag.simulate_recompile_storm = broken_simulate_recompile_storm

    try:
        out["catches_ignored_fallbacks"] = 0.0 if _survives(path) else 1.0
    finally:
        storm_mod.simulate_recompile_storm = original_fn
        compiler_diag.simulate_recompile_storm = original_fn

    return out
