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
        return None
    for fn in fns:
        fn()
    return True


def check(workdir):
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_buffer_overwrites": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        res = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if res is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import cudagraphs.memory as mem

    good_fix = mem.fix_buffer_overwrites

    def broken_fix_buffer_overwrites(operations, unsafe_aliases):
        return operations

    mem.fix_buffer_overwrites = broken_fix_buffer_overwrites
    import cudagraphs

    cudagraphs.memory.fix_buffer_overwrites = broken_fix_buffer_overwrites

    try:
        survived = _run(path)
        out["catches_buffer_overwrites"] = 0.0 if survived else 1.0
    except Exception:
        out["catches_buffer_overwrites"] = 1.0
    finally:
        mem.fix_buffer_overwrites = good_fix
        cudagraphs.memory.fix_buffer_overwrites = good_fix

    return out
