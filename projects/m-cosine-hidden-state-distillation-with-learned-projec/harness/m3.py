import importlib.util
import os
import numpy as np


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unnormalized_cosine": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import distill.hidden as dh
    orig_class = dh.LearnedProjectionCosineLoss

    class BrokenLearnedProjectionCosineLoss(orig_class):
        def forward(self, student_state: np.ndarray, teacher_state: np.ndarray) -> float:
            proj = student_state @ self.W + self.b
            cosine_sim = np.sum(proj * teacher_state, axis=-1)
            return float(np.mean(1.0 - cosine_sim))

    dh.LearnedProjectionCosineLoss = BrokenLearnedProjectionCosineLoss
    try:
        out["catches_unnormalized_cosine"] = 0.0 if _survives(path) else 1.0
    finally:
        dh.LearnedProjectionCosineLoss = orig_class

    return out
