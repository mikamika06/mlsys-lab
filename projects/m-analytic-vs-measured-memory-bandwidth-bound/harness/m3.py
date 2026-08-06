import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_test", path)
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_dtype_scaling_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Learner tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import bwbound.analytic as analytic_mod

    orig_func = analytic_mod.compute_analytic_bound

    def buggy_compute_analytic(tensors, flops, peak_bandwidth_gbps, peak_tflops):
        total_bytes = 0.0
        for shape, _ in tensors:
            num_elements = 1
            for dim in shape:
                num_elements *= dim
            total_bytes += num_elements * 4.0

        time_mem = total_bytes / (peak_bandwidth_gbps * 1e9)
        time_compute = flops / (peak_tflops * 1e12)
        analytic_time = max(time_mem, time_compute)
        arithmetic_intensity = flops / total_bytes if total_bytes > 0 else 0.0
        is_memory_bound = time_mem >= time_compute

        return {
            "total_bytes": total_bytes,
            "arithmetic_intensity": arithmetic_intensity,
            "time_mem_sec": time_mem,
            "time_compute_sec": time_compute,
            "analytic_time_sec": analytic_time,
            "is_memory_bound": is_memory_bound,
        }

    analytic_mod.compute_analytic_bound = buggy_compute_analytic
    try:
        survived = _survives(path)
        out["catches_dtype_scaling_bug"] = 0.0 if survived else 1.0
    finally:
        analytic_mod.compute_analytic_bound = orig_func

    return out
