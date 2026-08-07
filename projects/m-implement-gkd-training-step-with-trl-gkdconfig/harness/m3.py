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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_swapped_kl": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct reference implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import gkdstep.loss as l

    good_compute = l.compute_divergence

    def swapped_compute(teacher_logits, student_logits, divergence_type="forward_kl", temperature=1.0):
        if divergence_type == "forward_kl":
            swapped = "reverse_kl"
        elif divergence_type == "reverse_kl":
            swapped = "forward_kl"
        else:
            swapped = divergence_type
        return good_compute(teacher_logits, student_logits, divergence_type=swapped, temperature=temperature)

    l.compute_divergence = swapped_compute
    try:
        out["catches_swapped_kl"] = 0.0 if _survives(path) else 1.0
    finally:
        l.compute_divergence = good_compute

    return out
