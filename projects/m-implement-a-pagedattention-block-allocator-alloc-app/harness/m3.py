import importlib.util
import os

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("test_") and callable(getattr(mod, n))]
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_cow_failure": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct allocator: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import kv.allocator as a
    good_append = a.PagedAllocator.append

    def bad_append(self, seq_id, tokens):
        table = self.block_tables[seq_id]
        for t in tokens:
            if not table:
                b = self.free_blocks.pop()
                self.ref_counts[b] = 1
                table.append(b)
            last_b = table[-1]
            if len(self.blocks[last_b]) == self.block_size:
                b = self.free_blocks.pop()
                self.ref_counts[b] = 1
                table.append(b)
                last_b = b
            # Intentional bug: skipped CoW logic here
            self.blocks[last_b].append(t)

    a.PagedAllocator.append = bad_append
    try:
        out["catches_cow_failure"] = 0.0 if _survives(path) else 1.0
    finally:
        a.PagedAllocator.append = good_append

    return out
