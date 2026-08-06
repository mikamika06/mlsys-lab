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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_bad_hashing": 0.0,
        "catches_bad_affinity": 0.0
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import chash.router as r_mod
    import chash.affinity as a_mod

    orig_get_replica = r_mod.ConsistentHashRing.get_replica

    def modulo_get_replica(self, key):
        if not self.node_map:
            return None
        nodes = sorted(list(set(self.node_map.values())))
        idx = abs(hash(key)) % len(nodes)
        return nodes[idx]

    r_mod.ConsistentHashRing.get_replica = modulo_get_replica
    try:
        out["catches_bad_hashing"] = 0.0 if _survives(path) else 1.0
    finally:
        r_mod.ConsistentHashRing.get_replica = orig_get_replica

    orig_route = a_mod.SessionAffinityRouter.route

    def broken_route(self, session_id, key, current_time):
        return self.ring.get_replica(key)

    a_mod.SessionAffinityRouter.route = broken_route
    try:
        out["catches_bad_affinity"] = 0.0 if _survives(path) else 1.0
    finally:
        a_mod.SessionAffinityRouter.route = orig_route

    return out
