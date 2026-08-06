import sys
import os
import importlib.util


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_cache": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "test_regression.py is missing"
        return out

    sys.path.insert(0, workdir)
    try:
        def _run(target_path):
            spec = importlib.util.spec_from_file_location("test_regression", target_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
            if not fns:
                return None
            for fn in fns:
                fn()
            return True

        try:
            first = _run(path)
        except Exception as e:
            out["has_tests"] = 1.0
            out["_note"] = f"fails on good: {e}"
            return out

        if first is None:
            return out

        out["has_tests"] = 1.0
        out["passes_on_good"] = 1.0

        from mps_mem.mock_device import MPSDevice
        good_empty = MPSDevice.empty_cache
        MPSDevice.empty_cache = lambda self: None
        try:
            try:
                _run(path)
            except Exception:
                out["catches_broken_cache"] = 1.0
        finally:
            MPSDevice.empty_cache = good_empty
    finally:
        sys.path.pop(0)

    return out
