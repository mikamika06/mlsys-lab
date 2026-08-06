import importlib.util
import os


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
        for name in dir(mod):
            obj = getattr(mod, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, object)
                and any(m.startswith("test_") for m in dir(obj))
            ):
                suite = obj()
                for m in dir(obj):
                    if m.startswith("test_"):
                        getattr(suite, m)()
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
        "catches_invalid_moe": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on good: {e}"
        return out
    if first is None:
        out["_note"] = "no tests found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import ggufmap.moe as m_mod

    good = m_mod.build_moe_inventory

    def broken(tensor_names):
        return []

    m_mod.build_moe_inventory = broken
    import ggufmap

    ggufmap.moe.build_moe_inventory = broken
    try:
        out["catches_invalid_moe"] = 0.0 if _survives(path) else 1.0
    finally:
        m_mod.build_moe_inventory = good
        ggufmap.moe.build_moe_inventory = good
    return out
