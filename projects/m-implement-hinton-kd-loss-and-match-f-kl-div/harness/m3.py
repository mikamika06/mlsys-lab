import importlib.util
import os
import torch


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_missing_t2": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import distill.loss as dl
    good_fn = dl.hinton_kd_loss

    def broken_hinton_kd_loss(student_logits, teacher_logits, temperature=1.0, alpha=0.5, labels=None):
        import torch.nn.functional as F
        s_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
        t_soft = F.softmax(teacher_logits / temperature, dim=-1)
        kd_loss = F.kl_div(s_log_soft, t_soft, reduction='batchmean')
        if labels is None or alpha == 0.0:
            return kd_loss
        ce_loss = F.cross_entropy(student_logits, labels)
        return alpha * ce_loss + (1.0 - alpha) * kd_loss

    dl.hinton_kd_loss = broken_hinton_kd_loss
    try:
        survived = _survives(path)
        out["catches_missing_t2"] = 0.0 if survived else 1.0
        if survived:
            out["_note"] = "regression test failed to catch the missing T^2 scaling implementation"
    finally:
        dl.hinton_kd_loss = good_fn
    return out
