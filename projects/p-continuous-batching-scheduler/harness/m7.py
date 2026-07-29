import importlib.util
import os


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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0,
           "catches_unbounded_admission": 0.0, "catches_broken_refcount": 0.0,
           "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import sched.allocator as alloc
    import sched.policy as pol

    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_admit = pol.should_admit
    pol.should_admit = lambda state: True
    try:
        out["catches_unbounded_admission"] = 0.0 if _survives(path) else 1.0
    finally:
        pol.should_admit = good_admit

    good_release = alloc.Allocator.release

    def leaky(self, block):
        if block not in self.free:
            self.free.append(block)
            self.free.sort()

    alloc.Allocator.release = leaky
    try:
        out["catches_broken_refcount"] = 0.0 if _survives(path) else 1.0
    finally:
        alloc.Allocator.release = good_release

    out["faults_caught"] = out["catches_unbounded_admission"] + out["catches_broken_refcount"]
    return out
