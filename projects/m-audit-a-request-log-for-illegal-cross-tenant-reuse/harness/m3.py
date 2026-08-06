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


def check(workdir):
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_missing_tenant_check": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        res = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on correct code: {type(e).__name__}: {e}"
        return out

    if res is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import logaudit.tracker as tracker_mod

    orig_process = tracker_mod.CacheTracker.process_event

    def broken_process(self, event):
        event_type = event.get("type")
        tenant_id = event.get("tenant_id")
        block_id = event.get("block_id")

        if event_type == "allocate":
            tokens = event.get("tokens", [])
            self.block_owners[block_id] = tenant_id
            self.block_tokens[block_id] = list(tokens)
            return None

        if event_type == "lookup":
            return None
        return None

    tracker_mod.CacheTracker.process_event = broken_process
    try:
        test_passed = False
        try:
            test_passed = _run(path) is True
        except Exception:
            test_passed = False

        if not test_passed:
            out["catches_missing_tenant_check"] = 1.0
        else:
            out["_note"] = "Tests failed to catch a tracker that never detects lookups/violations"
    finally:
        tracker_mod.CacheTracker.process_event = orig_process

    return out
