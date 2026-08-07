import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
        "catches_broken_metrics": 0.0,
        "catches_broken_router": 0.0,
        "faults_caught": 0.0,
    }

    if not os.path.isfile(path):
        return out

    import moe.metrics as metrics
    import moe.router as router

    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    orig_measure = metrics.measure_distribution

    def broken_measure(selected_experts, num_experts):
        return np.ones(num_experts, dtype=np.int64)

    metrics.measure_distribution = broken_measure
    try:
        out["catches_broken_metrics"] = 0.0 if _survives(path) else 1.0
    finally:
        metrics.measure_distribution = orig_measure

    orig_route = router.MoERouter.route

    def broken_route(self, x, top_k=2):
        N = x.shape[0]
        probs = np.ones((N, self.num_experts)) / self.num_experts
        idxs = np.zeros((N, top_k), dtype=int)
        weights = np.ones((N, top_k)) / top_k
        return probs, idxs, weights

    router.MoERouter.route = broken_route
    try:
        out["catches_broken_router"] = 0.0 if _survives(path) else 1.0
    finally:
        router.MoERouter.route = orig_route

    out["faults_caught"] = (
        out["catches_broken_metrics"] + out["catches_broken_router"]
    )
    return out
