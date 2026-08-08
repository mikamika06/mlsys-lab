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


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_page_alignment_fault": 0.0,
        "catches_dedup_overflow_fault": 0.0,
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
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import memload.loader as l
    import memload.dedup as d

    orig_compare = l.compare_load_footprint
    orig_dedup = d.calculate_dedup_savings

    def broken_page_alignment(tensors, page_size=4096):
        res = orig_compare(tensors, page_size)
        if tensors and res["mmap_resident_bytes"] > 0:
            raw_acc = sum(t.get("accessed_bytes", t["size_bytes"]) for t in tensors)
            res["mmap_resident_bytes"] = raw_acc
        return res

    def broken_dedup_savings(tensors, page_size=4096):
        res = orig_dedup(tensors, page_size)
        if res["disk_savings_bytes"] > 0:
            res["disk_savings_bytes"] = res["disk_savings_bytes"] * 2
            res["heap_savings_bytes"] = res["heap_savings_bytes"] * 2
        return res

    l.compare_load_footprint = broken_page_alignment
    try:
        out["catches_page_alignment_fault"] = 0.0 if _survives(path) else 1.0
    finally:
        l.compare_load_footprint = orig_compare

    d.calculate_dedup_savings = broken_dedup_savings
    try:
        out["catches_dedup_overflow_fault"] = 0.0 if _survives(path) else 1.0
    finally:
        d.calculate_dedup_savings = orig_dedup

    return out
