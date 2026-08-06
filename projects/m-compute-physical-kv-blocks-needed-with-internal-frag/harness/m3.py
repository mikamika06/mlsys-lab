import importlib.util
import os
import ref


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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_tail_allocation_bug": 0.0}

    from paged_kv.startup import parse_vllm_startup_and_compute_capacity
    for i, log_text in enumerate(ref.SAMPLE_LOGS):
        want = ref.parse_vllm_startup_and_compute_capacity(log_text, seq_len=100, block_size=16, bytes_per_token=1024)
        got = parse_vllm_startup_and_compute_capacity(log_text, seq_len=100, block_size=16, bytes_per_token=1024)
        if got != want:
            out["_note"] = f"Startup parser mismatch on sample log {i}: got {got}, want {want}"
            return out

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"Learner tests failed on valid codebase: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import paged_kv.allocator as alloc_mod
    orig_alloc = alloc_mod.BlockTableAllocator.allocate

    def buggy_allocate(self, seq_id, initial_seq_len):
        needed_blocks = initial_seq_len // self.block_size
        if len(self.free_pool) < needed_blocks:
            raise MemoryError("Out of physical KV blocks")
        allocated = [self.free_pool.pop(0) for _ in range(needed_blocks)]
        self.tables[seq_id] = allocated
        self.seq_lens[seq_id] = initial_seq_len
        return list(allocated)

    alloc_mod.BlockTableAllocator.allocate = buggy_allocate
    try:
        failed = not _survives(path)
        out["catches_tail_allocation_bug"] = 1.0 if failed else 0.0
    finally:
        alloc_mod.BlockTableAllocator.allocate = orig_alloc

    return out
