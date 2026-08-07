import importlib.util
import os
import sys
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

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_ignored_launch_overhead": 0.0}
    sys.path.insert(0, workdir)
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests failed on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import launchbound.profiler as profiler
    good_predict = profiler.predict_small_batch_speedup

    def faulty_predict(op_count, baseline_batch_size, target_batch_size, cpu_launch_overhead_us, gpu_time_per_op_per_batch_us):
        baseline_gpu_time = op_count * baseline_batch_size * gpu_time_per_op_per_batch_us
        target_gpu_time = op_count * target_batch_size * gpu_time_per_op_per_batch_us
        return baseline_gpu_time / target_gpu_time if target_gpu_time > 0 else 1.0

    profiler.predict_small_batch_speedup = faulty_predict
    try:
        catches = False
        try:
            _run(path)
        except Exception:
            catches = True
        out["catches_ignored_launch_overhead"] = 1.0 if catches else 0.0
    finally:
        profiler.predict_small_batch_speedup = good_predict

    return out
