import os
import importlib.util

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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_refcount": 0.0}

    if not os.path.isfile(path):
        return out

    import kv.allocator as alloc

    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    orig_fork = alloc.KVAllocator.fork_sequence

    def bad_fork(self, parent_id):
        sid = orig_fork(self, parent_id)
        for b in self.get_block_table(sid):
            self.ref_counts[b] -= 1
        return sid

    alloc.KVAllocator.fork_sequence = bad_fork
    try:
        if not _survives(path):
            out["catches_broken_refcount"] = 1.0
    finally:
        alloc.KVAllocator.fork_sequence = orig_fork

    return out
