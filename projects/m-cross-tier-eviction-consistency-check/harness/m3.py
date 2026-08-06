"""Milestone 3 safeguard harness check."""

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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_invalidation": 0.0}
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

    import eviction.manager as em
    good_evict = em.CrossTierEvictionManager.evict_from_t0

    def faulty_evict(self, block_id, preserve_in_t1=True):
        if block_id not in self.t0_state:
            return False
        self.t0_state.pop(block_id)
        # Bug: ignore preserve_in_t1=False and always keep in T1
        if block_id not in self.t1_state:
            self.t1_state[block_id] = {"hash": "stale_data", "stale": True, "status": "VALID"}
        return True

    em.CrossTierEvictionManager.evict_from_t0 = faulty_evict
    try:
        out["catches_ignored_invalidation"] = 0.0 if _survives(path) else 1.0
    finally:
        em.CrossTierEvictionManager.evict_from_t0 = good_evict

    return out
