import importlib.util
import os
import numpy as np
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "argmin_index": 0.0}
    
    try:
        from fp4quant.sweep import sweep_block_size
        np.random.seed(789)
        x = np.random.randn(256)
        block_sizes = [16, 32, 64, 128]
        want_idx, want_err = ref.reference_sweep(x, block_sizes)
        got_idx, got_err = sweep_block_size(x, block_sizes)
        if got_idx == want_idx and np.isclose(got_err, want_err):
            out["argmin_index"] = 1.0
        else:
            out["_note"] = f"sweep_block_size mismatch. Got ({got_idx}, {got_err}), expected ({want_idx}, {want_err})"
            return out
    except Exception as e:
        out["_note"] = f"sweep_block_size failed: {e}"
        return out

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"The tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import fp4quant.sweep as s
    good_sweep = s.sweep_block_size

    def broken_sweep(x, block_sizes):
        idx, err = good_sweep(x, block_sizes)
        return (idx + 1) % len(block_sizes), err

    s.sweep_block_size = broken_sweep
    import fp4quant
    fp4quant.sweep.sweep_block_size = broken_sweep

    try:
        catches_broken = not _survives(path)
        if not catches_broken:
            out["_note"] = "tests/test_regression.py failed to catch broken argmin index selection"
            out["argmin_index"] = 0.0
    finally:
        s.sweep_block_size = good_sweep
        fp4quant.sweep.sweep_block_size = good_sweep

    return out
