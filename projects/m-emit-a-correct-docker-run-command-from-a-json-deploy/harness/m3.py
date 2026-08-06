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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_shm_mount": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import deploy.k8s as k8s_mod
    good_func = k8s_mod.docker_to_k8s

    def broken_k8s(spec):
        res = good_func(spec)
        res["spec"]["volumes"] = [v for v in res["spec"].get("volumes", []) if v.get("name") != "dshm"]
        res["spec"]["containers"][0]["volumeMounts"] = [
            vm for vm in res["spec"]["containers"][0].get("volumeMounts", []) if vm.get("name") != "dshm"
        ]
        return res

    k8s_mod.docker_to_k8s = broken_k8s
    try:
        out["catches_broken_shm_mount"] = 0.0 if _survives(path) else 1.0
    finally:
        k8s_mod.docker_to_k8s = good_func
    return out
