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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_padding": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct parser: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import gguf_parser.parser as p
    good = p.compute_overhead

    def broken(manifest):
        header_end = manifest["header_end_offset"]
        meta_end = manifest["_meta_end"]
        alignment = manifest.get("metadata", {}).get("general.alignment", 32)
        # BUG: returns alignment instead of 0 when perfectly aligned
        padding = alignment - (header_end % alignment)

        return {
            "metadata_bytes": meta_end - 24,
            "tensor_info_bytes": header_end - meta_end,
            "padding_waste": padding
        }

    p.compute_overhead = broken
    if 'gguf_parser.parser' in sys.modules:
        sys.modules['gguf_parser.parser'].compute_overhead = broken

    try:
        out["catches_broken_padding"] = 0.0 if _survives(path) else 1.0
    finally:
        p.compute_overhead = good
        if 'gguf_parser.parser' in sys.modules:
            sys.modules['gguf_parser.parser'].compute_overhead = good

    return out
