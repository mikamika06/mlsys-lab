import importlib.util
import os
import ref


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
           "catches_bad_affinity": 0.0, "catches_remote_alloc": 0.0,
           "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import numa_tuning.affinity as aff

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail: {type(e).__name__}: {e}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_pinning = aff.apply_pinning

    def bad_pinning(thread_id, core_id):
        return {"thread": thread_id, "pinned_core": -1, "status": "failed"}

    aff.apply_pinning = bad_pinning
    try:
        out["catches_bad_affinity"] = 0.0 if _survives(path) else 1.0
    finally:
        aff.apply_pinning = good_pinning

    good_alloc = aff.allocate_numa_memory

    def bad_alloc(size, node_id):
        return {"size": size, "node": node_id, "allocated": False}

    aff.allocate_numa_memory = bad_alloc
    try:
        out["catches_remote_alloc"] = 0.0 if _survives(path) else 1.0
    finally:
        aff.allocate_numa_memory = good_alloc

    out["faults_caught"] = out["catches_bad_affinity"] + out["catches_remote_alloc"]
    return out
