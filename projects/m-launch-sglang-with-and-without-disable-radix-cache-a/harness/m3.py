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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_cache": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import sgl_utils.launcher as l_mod
    good_builder = l_mod.build_launch_command

    def broken_builder(model_path, port=30000, disable_radix_cache=False, extra_args=None):
        cmd = ["python", "-m", "sglang.launch_server", "--model-path", str(model_path), "--port", str(port)]
        if not disable_radix_cache:
            cmd.append("--disable-radix-cache")
        return cmd

    l_mod.build_launch_command = broken_builder
    try:
        out["catches_broken_cache"] = 0.0 if _survives(path) else 1.0
    finally:
        l_mod.build_launch_command = good_builder
    return out
