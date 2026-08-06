import importlib.util
import os
import sys

def _run(path):
    spec = importlib.util.spec_from_file_location("test_regression", path)
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_missing_offset": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        return out

    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    try:
        import kvblocks.mapping as m
    except ImportError:
        return out

    good = m.gather_slot_mapping

    def bad_gather(seq_lens, block_tables, block_size):
        slots = []
        for l, table in zip(seq_lens, block_tables):
            for i in range(l):
                # INTENTIONALLY BROKEN: drops the + (i % block_size) offset
                slots.append(table[i // block_size] * block_size)
        return slots

    m.gather_slot_mapping = bad_gather

    try:
        out["catches_missing_offset"] = 0.0 if _survives(path) else 1.0
    finally:
        m.gather_slot_mapping = good

    return out
