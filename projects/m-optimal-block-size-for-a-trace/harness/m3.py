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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_invalid_allocation": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on reference implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import kvblock.triage as t
    good_triage = t.triage_block_table

    def buggy_triage(block_table, total_seq_len, block_size, max_valid_block_id):
        expected_blocks = (total_seq_len + block_size - 1) // block_size if total_seq_len > 0 else 0
        return {
            "is_valid": True,
            "issues": [],
            "expected_blocks": expected_blocks,
            "repaired_table": list(block_table),
        }

    t.triage_block_table = buggy_triage
    import kvblock
    kvblock.triage.triage_block_table = buggy_triage

    try:
        out["catches_invalid_allocation"] = 0.0 if _survives(path) else 1.0
    finally:
        t.triage_block_table = good_triage
        kvblock.triage.triage_block_table = good_triage

    return out
