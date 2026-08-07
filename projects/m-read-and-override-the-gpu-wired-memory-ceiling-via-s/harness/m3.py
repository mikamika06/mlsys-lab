import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_test", path)
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unsafe_overrides": 0.0}
    test_path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(test_path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    try:
        run_res = _run(test_path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on valid reference implementation: {e}"
        return out

    if run_res is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import sysctl_mem.ceiling as ceiling_mod
    original_fn = ceiling_mod.generate_sysctl_override

    def faulty_override(memsize_bytes, target_percentage):
        total_mb = memsize_bytes // (1024 * 1024)
        target_mb = int(total_mb * (target_percentage / 100.0))
        return f"sudo sysctl iogpu.wired_mem_limit_mb={target_mb}"

    ceiling_mod.generate_sysctl_override = faulty_override

    try:
        survived = _survives(test_path)
        out["catches_unsafe_overrides"] = 0.0 if survived else 1.0
        if survived:
            out["_note"] = "Learner tests passed even when bounds check on target_percentage was removed"
    finally:
        ceiling_mod.generate_sysctl_override = original_fn

    return out
