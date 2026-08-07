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
        "catches_unmasked_gradient": 0.0,
        "catches_budget_overrun": 0.0,
        "faults_caught": 0.0,
    }
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import heal.mask as mask_mod
    import heal.trainer as trainer_mod

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

    orig_mask_grads = mask_mod.MaskManager.mask_gradients

    def leaky_mask_grads(self, grads):
        pass

    mask_mod.MaskManager.mask_gradients = leaky_mask_grads
    try:
        out["catches_unmasked_gradient"] = 0.0 if _survives(path) else 1.0
    finally:
        mask_mod.MaskManager.mask_gradients = orig_mask_grads

    orig_step = trainer_mod.HealerTrainer.step

    def leaky_step(self, X_batch, y_batch):
        loss, weight_grads, bias_grads = self.model.forward_backward(X_batch, y_batch)
        self.mask_mgr.mask_gradients(weight_grads)
        self.model.apply_gradients(weight_grads, bias_grads, self.lr)
        self.mask_mgr.apply_mask()
        self.step_count += 1
        self.history.append(float(loss))
        return float(loss)

    trainer_mod.HealerTrainer.step = leaky_step
    try:
        out["catches_budget_overrun"] = 0.0 if _survives(path) else 1.0
    finally:
        trainer_mod.HealerTrainer.step = orig_step

    out["faults_caught"] = out["catches_unmasked_gradient"] + out["catches_budget_overrun"]
    return out
