import importlib.util
import os
import ref

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
    from serving.simulator import simulate_tiering

    out = {
        "gpu_hit_rel_err": 0.0,
        "host_hit_rel_err": 0.0,
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_broken_tiering": 0.0
    }

    for reqs in ref.WORKLOADS:
        gc, hc = 10, 10
        w_gpu, w_host = ref.simulate_tiering(reqs, gc, hc)
        g_gpu, g_host = simulate_tiering(reqs, gc, hc)

        out["gpu_hit_rel_err"] = max(out["gpu_hit_rel_err"], abs(g_gpu - w_gpu) / (w_gpu + 1e-9))
        out["host_hit_rel_err"] = max(out["host_hit_rel_err"], abs(g_host - w_host) / (w_host + 1e-9))

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct simulator: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import serving.simulator as sim
    good_tiering = sim.simulate_tiering

    def broken_tiering(reqs, gpu_c, host_c):
        return good_tiering(reqs, gpu_c, 0)

    sim.simulate_tiering = broken_tiering
    try:
        out["catches_broken_tiering"] = 0.0 if _survives(path) else 1.0
    finally:
        sim.simulate_tiering = good_tiering

    return out
