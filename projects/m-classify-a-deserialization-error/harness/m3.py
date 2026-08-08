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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_missing_arch_check": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct plan: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import plan
    good = plan.diagnose_load

    def bad_diagnose(engine_bytes, env_trt, env_sm, env_os):
        if len(engine_bytes) < 20:
            return {"status": "ERR_TRUNCATED", "penalty": 0.0}
        h = plan.parse_header(engine_bytes)
        if h["magic"] != b'TRT\x00': return {"status": "ERR_MAGIC", "penalty": 0.0}
        if h["trt_version"] != env_trt: return {"status": "ERR_TRT_VERSION", "penalty": 0.0}
        if h["os_id"] != env_os: return {"status": "ERR_OS", "penalty": 0.0}

        if h["build_sm"] != env_sm:
            if h["hw_compat"] == 0:
                return {"status": "ERR_SM_ARCH", "penalty": 0.0}
            else:
                # INJECTED FAULT: Missing the `< 80` SM check for hardware compatibility
                return {"status": "OK", "penalty": 8.5}
        else:
            if h["hw_compat"] == 1:
                return {"status": "OK", "penalty": 3.0}
            return {"status": "OK", "penalty": 0.0}

    plan.diagnose_load = bad_diagnose
    try:
        out["catches_missing_arch_check"] = 0.0 if _survives(path) else 1.0
    finally:
        plan.diagnose_load = good

    return out
