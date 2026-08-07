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
        "catches_middle_loss": 0.0,
        "catches_attention_decay": 0.0,
        "faults_caught": 0.0
    }
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import long_ctx.generator as gen
    import long_ctx.evaluator as eval_mod

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_gen = gen.generate_context_with_fact
    gen.generate_context_with_fact = lambda length, pos, fact: "Corrupted Context Without Fact"
    try:
        out["catches_middle_loss"] = 0.0 if _survives(path) else 1.0
    finally:
        gen.generate_context_with_fact = good_gen

    good_eval = eval_mod.evaluate_position_curve
    eval_mod.evaluate_position_curve = lambda model, ctxs: {k: 0.0 for k in ctxs}
    try:
        out["catches_attention_decay"] = 0.0 if _survives(path) else 1.0
    finally:
        eval_mod.evaluate_position_curve = good_eval

    out["faults_caught"] = out["catches_middle_loss"] + out["catches_attention_decay"]
    return out
