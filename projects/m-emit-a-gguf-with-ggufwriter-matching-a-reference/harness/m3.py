import importlib.util
import os
import sys
import ref

def _run(path: str) -> bool:
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True

def check(workdir: str) -> dict:
    sys.path.insert(0, workdir)
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_corrupted_metadata": 0.0}
    test_path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(test_path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        res = _run(test_path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if res is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import ggftool.patch as p
    orig_patch = p.patch_metadata_in_place

    def corrupted_patch(gguf_bytes: bytes, patches: dict[str, str]) -> bytes:
        res = bytearray(orig_patch(gguf_bytes, patches))
        if len(res) > 32:
            res[-1] ^= 0xFF
        return bytes(res)

    p.patch_metadata_in_place = corrupted_patch
    try:
        caught = False
        try:
            _run(test_path)
        except Exception:
            caught = True
        out["catches_corrupted_metadata"] = 1.0 if caught else 0.0
    finally:
        p.patch_metadata_in_place = orig_patch

    return out
