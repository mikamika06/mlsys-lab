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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_fragmentation_bug": 0.0}
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

    import kvcache.allocator as a
    original_init = a.BlockAllocator.__init__
    original_allocate = a.BlockAllocator.allocate
    original_free = a.BlockAllocator.free

    class BrokenAllocator:
        def __init__(self, num_blocks: int):
            self.num_blocks = num_blocks
            self.head = 0

        def allocate(self) -> int:
            if self.head >= self.num_blocks:
                raise MemoryError()
            res = self.head
            self.head += 1
            return res

        def free(self, block_id: int):
            if block_id == self.head - 1:
                self.head -= 1

    a.BlockAllocator.__init__ = BrokenAllocator.__init__
    a.BlockAllocator.allocate = BrokenAllocator.allocate
    a.BlockAllocator.free = BrokenAllocator.free

    try:
        survived = _survives(path)
        out["catches_fragmentation_bug"] = 0.0 if survived else 1.0
    finally:
        a.BlockAllocator.__init__ = original_init
        a.BlockAllocator.allocate = original_allocate
        a.BlockAllocator.free = original_free

    return out
