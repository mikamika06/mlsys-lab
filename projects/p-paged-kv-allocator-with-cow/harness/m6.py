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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_fork": 0.0}
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    import kv.allocator as alloc

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"fails on good: {e}"
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_fork = alloc.PagedKVAllocator.fork_seq
    def bad_fork(self, parent_id, child_id):
        if child_id in self.block_tables:
            raise ValueError("Child already exists")
        self.block_tables[child_id] = list(self.block_tables[parent_id])
        self.seq_lengths[child_id] = self.seq_lengths[parent_id]

    alloc.PagedKVAllocator.fork_seq = bad_fork
    try:
        out["catches_bad_fork"] = 0.0 if _survives(path) else 1.0
    finally:
        alloc.PagedKVAllocator.fork_seq = good_fork

    return out
