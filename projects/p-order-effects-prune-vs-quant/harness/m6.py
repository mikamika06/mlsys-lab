import os
import importlib.util

def _run(path):
    spec = importlib.util.spec_from_file_location("test_regression", path)
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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "faults_caught": 0.0}
    if not os.path.isfile(path):
        return out

    try:
        from compression import pipeline
    except ImportError:
        return out

    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0

    good_joint = pipeline.joint_recipe

    def leaky_joint1(w, p, b):
        from compression.ops import prune, quantize
        return quantize(prune(w, p), b)

    pipeline.joint_recipe = leaky_joint1
    if not _survives(path):
        out["faults_caught"] += 1.0

    def leaky_joint2(w, p, b):
        return w * 0.0

    pipeline.joint_recipe = leaky_joint2
    if not _survives(path):
        out["faults_caught"] += 1.0

    pipeline.joint_recipe = good_joint
    return out
