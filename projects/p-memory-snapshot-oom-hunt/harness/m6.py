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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_retention": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        return out
    out["has_tests"] = 1.0

    try:
        first = _run(path)
    except Exception:
        return out
    if first is None:
        return out
    out["passes_on_good"] = 1.0

    import leak.detector as detector

    good_fix = detector.MemorySnapshotAnalyzer.fix_retention

    def broken_fix(self):
        self.fixed = False
        return False

    detector.MemorySnapshotAnalyzer.fix_retention = broken_fix
    try:
        out["catches_broken_retention"] = 0.0 if _survives(path) else 1.0
    finally:
        detector.MemorySnapshotAnalyzer.fix_retention = good_fix

    out["faults_caught"] = out["catches_broken_retention"]
    return out
