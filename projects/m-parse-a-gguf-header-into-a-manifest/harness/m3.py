import importlib.util
import os
import ref


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


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_unaligned_waste": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"The tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        )
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import gguf_parser.overhead as ov

    good_func = ov.compute_container_overhead

    def broken_compute_container_overhead(data: bytes) -> dict:
        res = good_func(data)
        res["alignment_waste"] = 0
        res["total_overhead"] = res["data_offset"]
        return res

    ov.compute_container_overhead = broken_compute_container_overhead
    import gguf_parser

    gguf_parser.compute_container_overhead = broken_compute_container_overhead

    try:
        out["catches_unaligned_waste"] = 0.0 if _survives(path) else 1.0
    finally:
        ov.compute_container_overhead = good_func
        gguf_parser.compute_container_overhead = good_func

    return out
