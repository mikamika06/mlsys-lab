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
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_underlaunched_grid": 0.0,
        "catches_unmasked_boundary": 0.0,
    }

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import vadd.grid as g
    import vadd.kernel as k

    orig_grid_fn = g.get_grid_num_programs
    orig_k_grid_fn = getattr(k, "get_grid_num_programs", orig_grid_fn)
    orig_step_fn = k.vector_add_kernel_step

    def underlaunched_grid(n, block_size):
        return n // block_size

    g.get_grid_num_programs = underlaunched_grid
    if hasattr(k, "get_grid_num_programs"):
        k.get_grid_num_programs = underlaunched_grid

    try:
        fault1_survived = _survives(path)
        out["catches_underlaunched_grid"] = 0.0 if fault1_survived else 1.0
    finally:
        g.get_grid_num_programs = orig_grid_fn
        if hasattr(k, "get_grid_num_programs"):
            k.get_grid_num_programs = orig_k_grid_fn

    def unmasked_step(x, y, out_arr, pid, block_size, mask_boundary=True):
        offsets = pid * block_size + range(block_size)
        for idx in offsets:
            if idx < len(x):
                if idx >= (len(x) // block_size) * block_size:
                    pass
                else:
                    out_arr[idx] = x[idx] + y[idx]

    k.vector_add_kernel_step = unmasked_step
    try:
        fault2_survived = _survives(path)
        out["catches_unmasked_boundary"] = 0.0 if fault2_survived else 1.0
    finally:
        k.vector_add_kernel_step = orig_step_fn

    return out
