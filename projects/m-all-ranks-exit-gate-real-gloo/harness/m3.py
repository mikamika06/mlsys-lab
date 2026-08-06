import importlib.util
import os
import sys

def _run(path, workdir):
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

def _survives(path, workdir):
    try:
        return _run(path, workdir) is True
    except Exception:
        return False

def check(workdir):
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_blind_trust": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
        
    try:
        first = _run(path, workdir)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct plan: {type(e).__name__}: {str(e)[:120]}"
        return out
        
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
        
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import ft.checkpoint as chk
    good = chk.get_safe_resume_checkpoint

    def blind_trust(checkpoint_dir: str):
        latest_path = os.path.join(checkpoint_dir, "latest")
        if os.path.isfile(latest_path):
            with open(latest_path, "r") as f:
                return os.path.join(checkpoint_dir, f.read().strip())
        return None

    chk.get_safe_resume_checkpoint = blind_trust
    try:
        survived = _survives(path, workdir)
        out["catches_blind_trust"] = 0.0 if survived else 1.0
    finally:
        chk.get_safe_resume_checkpoint = good
        
    return out
