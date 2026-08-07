import importlib.util
import os


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unamortized_scales": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py missing"
        return out

    try:
        res = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if res is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import kquant.amortization as ka
    orig_fp = ka.compute_superblock_footprint

    def broken_fp(num_elements, superblock_size, subblock_size, quant_bits, scale_bits, super_scale_bits):
        res = orig_fp(num_elements, superblock_size, subblock_size, quant_bits, scale_bits, super_scale_bits)
        res["metadata_ratio"] = 1.0
        return res

    ka.compute_superblock_footprint = broken_fp

    try:
        try:
            _run(path)
            out["catches_unamortized_scales"] = 0.0
        except Exception:
            out["catches_unamortized_scales"] = 1.0
    finally:
        ka.compute_superblock_footprint = orig_fp

    return out
