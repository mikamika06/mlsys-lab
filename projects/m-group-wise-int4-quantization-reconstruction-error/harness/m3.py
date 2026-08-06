import importlib.util
import os
import numpy as np


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_global_scale_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import sys
    sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import quant.group_int4 as g
    good_quant = g.quantize_dequantize_int4
    good_mse = g.compute_reconstruction_mse

    def faulty_quantize_dequantize_int4(x: np.ndarray, group_size: int):
        orig_shape = x.shape
        flat_x = x.astype(np.float64).reshape(-1)
        scale = np.max(np.abs(flat_x)) / 7.0
        if scale == 0:
            scale = 1.0
        q = np.clip(np.round(flat_x / scale), -7, 7).astype(np.int8)
        dequant = (q.astype(np.float64) * scale).reshape(orig_shape)
        num_groups = flat_x.shape[0] // group_size
        scales = np.full(num_groups, scale)
        return q.reshape(orig_shape), dequant, scales

    def faulty_compute_reconstruction_mse(x: np.ndarray, group_size: int):
        _, dequant, _ = faulty_quantize_dequantize_int4(x, group_size)
        flat_orig = x.astype(np.float64).reshape(-1)
        flat_dequant = dequant.reshape(-1)
        diff = flat_orig - flat_dequant
        total_mse = float(np.mean(diff ** 2))
        num_groups = flat_orig.shape[0] // group_size
        group_diff = diff.reshape(num_groups, group_size)
        group_mse = np.mean(group_diff ** 2, axis=1).tolist()
        return {"total_mse": total_mse, "group_mse": group_mse}

    g.quantize_dequantize_int4 = faulty_quantize_dequantize_int4
    g.compute_reconstruction_mse = faulty_compute_reconstruction_mse

    try:
        out["catches_global_scale_bug"] = 0.0 if _survives(path) else 1.0
    finally:
        g.quantize_dequantize_int4 = good_quant
        g.compute_reconstruction_mse = good_mse

    return out
