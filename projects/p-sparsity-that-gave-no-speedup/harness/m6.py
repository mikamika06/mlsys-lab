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

    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_uncompressed_save": 0.0,
        "catches_corrupted_metadata": 0.0,
        "faults_caught": 0.0,
    }

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import sparse_eval.checkpoint as ckpt
    import sparse_eval.pattern as pat

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_save = ckpt.save_sparse_checkpoint

    def uncompressed_save(matrix):
        return {
            "compressed_weights": matrix,
            "metadata": matrix,
            "original_shape": matrix.shape,
            "format": "uncompressed_fake",
        }

    ckpt.save_sparse_checkpoint = uncompressed_save
    try:
        out["catches_uncompressed_save"] = 0.0 if _survives(path) else 1.0
    finally:
        ckpt.save_sparse_checkpoint = good_save

    good_compress = pat.compress_24_matrix

    def broken_compress(matrix):
        cw, meta = good_compress(matrix)
        meta = meta * 0
        return cw, meta

    pat.compress_24_matrix = broken_compress
    try:
        out["catches_corrupted_metadata"] = 0.0 if _survives(path) else 1.0
    finally:
        pat.compress_24_matrix = good_compress

    out["faults_caught"] = out["catches_uncompressed_save"] + out["catches_corrupted_metadata"]
    return out
