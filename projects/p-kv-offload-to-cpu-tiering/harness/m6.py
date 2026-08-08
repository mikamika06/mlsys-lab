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
           "catches_overflow_bypass": 0.0, "catches_cost_inversion": 0.0,
           "faults_caught": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import sys
    sys.path.insert(0, workdir)
    import kvtier.tier as tier_mod
    import kvtier.cost as cost_mod

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_offload = tier_mod.TierManager.offload

    def leaky_offload(self, session_id, blocks):
        self.storage[session_id] = blocks
        self.used_blocks += len(blocks)
        return True

    tier_mod.TierManager.offload = leaky_offload
    try:
        out["catches_overflow_bypass"] = 0.0 if _survives(path) else 1.0
    finally:
        tier_mod.TierManager.offload = good_offload

    good_cost = cost_mod.estimate_transfer_cost

    def bad_cost(num_tokens, bytes_per_token, bandwidth_gbps):
        return -1.0

    cost_mod.estimate_transfer_cost = bad_cost
    try:
        out["catches_cost_inversion"] = 0.0 if _survives(path) else 1.0
    finally:
        cost_mod.estimate_transfer_cost = good_cost

    out["faults_caught"] = out["catches_overflow_bypass"] + out["catches_cost_inversion"]
    return out
