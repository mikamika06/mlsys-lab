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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_broken_fusion": 0.0,
        "catches_invalid_parity": 0.0,
        "faults_caught": 0.0
    }
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import opt.transformer_fusion as tf
    import opt.validator as val

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on good implementation: {type(e).__name__}: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_fusion = tf.apply_transformer_fusion
    tf.apply_transformer_fusion = lambda g: {"nodes": g.get("nodes", []), "fused_count": 0}
    try:
        out["catches_broken_fusion"] = 0.0 if _survives(path) else 1.0
    finally:
        tf.apply_transformer_fusion = good_fusion

    good_parity = val.check_parity
    val.check_parity = lambda g1, g2, inp: {"max_diff": 0.5, "cosine_sim": 0.5, "parity_ok": 0}
    try:
        out["catches_invalid_parity"] = 0.0 if _survives(path) else 1.0
    finally:
        val.check_parity = good_parity

    out["faults_caught"] = out["catches_broken_fusion"] + out["catches_invalid_parity"]
    return out
