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
        "catches_positive_feedback_bug": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on valid reference implementation: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import moe_balance.bias_sim as bsim

    good_sim = bsim.simulate_deepseek_v3_bias_updates

    # Injected bug: invert the bias update error sign (positive feedback loop)
    def broken_sim(logits_batch_sequence, gamma=0.1, top_k=2):
        res = good_sim(logits_batch_sequence, gamma=gamma, top_k=top_k)
        if len(res["biases"]) > 0:
            res["biases"] = -res["biases"]  # Invert sign to create positive feedback
        return res

    bsim.simulate_deepseek_v3_bias_updates = broken_sim
    try:
        out["catches_positive_feedback_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        bsim.simulate_deepseek_v3_bias_updates = good_sim

    return out
