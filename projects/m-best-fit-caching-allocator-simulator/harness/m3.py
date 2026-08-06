import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_first_fit": 0.0,
        "catches_no_coalesce": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"Tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        )
        return out

    if first is None:
        out["_note"] = "No test_* functions found in test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import allocator.simulator as sim

    orig_cls = sim.CachingAllocator

    class FirstFitAllocator(orig_cls):
        def malloc(self, size: int) -> int:
            if size <= 0:
                raise ValueError("Allocation size must be positive")
            first_block_idx = None
            for i, block in enumerate(self.blocks):
                if not block.is_allocated and block.size >= size:
                    first_block_idx = i
                    break
            if first_block_idx is not None:
                block = self.blocks[first_block_idx]
                if block.size == size:
                    block.is_allocated = True
                    handle = self.next_handle
                    self.next_handle += 1
                    self.handles[handle] = block
                else:
                    rem_size = block.size - size
                    block.size = size
                    block.is_allocated = True
                    handle = self.next_handle
                    self.next_handle += 1
                    self.handles[handle] = block
                    rem_block = sim.Block(
                        block.addr + size, rem_size, False, block.segment_id
                    )
                    self.blocks.insert(first_block_idx + 1, rem_block)
            else:
                seg_size = max(self.default_segment_size, size)
                seg_id = len(self.segments)
                self.segments.append((self.next_addr, seg_size))
                self.reserved_deltas.append(seg_size)
                self.current_reserved += seg_size
                if seg_size == size:
                    block = sim.Block(self.next_addr, size, True, seg_id)
                    self.blocks.append(block)
                    self.next_addr += seg_size
                    handle = self.next_handle
                    self.next_handle += 1
                    self.handles[handle] = block
                else:
                    block = sim.Block(self.next_addr, size, True, seg_id)
                    rem_block = sim.Block(
                        self.next_addr + size, seg_size - size, False, seg_id
                    )
                    self.blocks.extend([block, rem_block])
                    self.next_addr += seg_size
                    handle = self.next_handle
                    self.next_handle += 1
                    self.handles[handle] = block

            self.current_allocated += size
            if self.current_allocated > self.peak_allocated:
                self.peak_allocated = self.current_allocated
            if self.current_reserved > self.peak_reserved:
                self.peak_reserved = self.current_reserved
            frag = self.current_reserved - self.current_allocated
            if frag > self.peak_fragmentation:
                self.peak_fragmentation = frag
            return handle

    class NoCoalesceAllocator(orig_cls):
        def coalesce(self) -> None:
            pass

    try:
        sim.CachingAllocator = FirstFitAllocator
        out["catches_first_fit"] = 0.0 if _survives(path) else 1.0

        sim.CachingAllocator = NoCoalesceAllocator
        out["catches_no_coalesce"] = 0.0 if _survives(path) else 1.0
    finally:
        sim.CachingAllocator = orig_cls

    return out
