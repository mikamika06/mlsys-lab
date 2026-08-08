import importlib.util
import os
import sys

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("test_") and callable(getattr(mod, n))]
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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0,
           "catches_attention_fault": 0.0, "catches_generator_fault": 0.0,
           "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import longctx.generator as gen
    import longctx.evaluator as ev

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_eval = ev.evaluate_curve
    def bad_eval(tasks, model_type="flawed"):
        return [{"position": t["position"], "accuracy": 1.0} for t in tasks]

    ev.evaluate_curve = bad_eval
    try:
        out["catches_attention_fault"] = 0.0 if _survives(path) else 1.0
    finally:
        ev.evaluate_curve = good_eval

    good_gen = gen.generate_tasks
    def bad_gen(context_len, num_positions, needle=""):
        return []

    gen.generate_tasks = bad_gen
    try:
        out["catches_generator_fault"] = 0.0 if _survives(path) else 1.0
    finally:
        gen.generate_tasks = good_gen

    out["faults_caught"] = out["catches_attention_fault"] + out["catches_generator_fault"]
    return out
