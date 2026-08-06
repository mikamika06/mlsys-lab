import importlib.util
import os
import numpy as np


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_shifted_logits": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import ppl.metrics as m
    import ppl.quant_eval as qe

    good_metrics = m.compute_logit_metrics

    def broken_metrics(base_logits, quant_logits):
        rolled = np.roll(quant_logits, 1, axis=0)
        return good_metrics(base_logits, rolled)

    m.compute_logit_metrics = broken_metrics
    qe.compute_logit_metrics = broken_metrics

    try:
        out["catches_shifted_logits"] = 0.0 if _survives(path) else 1.0
    finally:
        m.compute_logit_metrics = good_metrics
        qe.compute_logit_metrics = good_metrics

    return out
