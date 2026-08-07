import importlib.util
import os
import sys


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
    sys.path.insert(0, workdir)
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_faulty_replicas": 0.0,
        "catches_faulty_cost": 0.0,
        "faults_caught": 0.0,
    }

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    import capacity.planner as planner

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on reference implementation: {e}"
        return out

    if first is None:
        out["_note"] = "No test_ functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    orig_replicas = planner.compute_required_replicas
    planner.compute_required_replicas = lambda target_rps, single_replica_capacity, headroom_factor=1.2: 1
    try:
        out["catches_faulty_replicas"] = 0.0 if _survives(path) else 1.0
    finally:
        planner.compute_required_replicas = orig_replicas

    orig_cost = planner.compute_cost_per_million_tokens
    planner.compute_cost_per_million_tokens = (
        lambda replica_count, hourly_cost_per_replica, rps, avg_output_tokens: 0.0
    )
    try:
        out["catches_faulty_cost"] = 0.0 if _survives(path) else 1.0
    finally:
        planner.compute_cost_per_million_tokens = orig_cost

    out["faults_caught"] = out["catches_faulty_replicas"] + out["catches_faulty_cost"]
    return out
