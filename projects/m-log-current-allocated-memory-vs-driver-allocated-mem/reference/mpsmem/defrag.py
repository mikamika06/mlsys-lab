class SimulatedMPSAllocator:
    def __init__(self, driver_limit):
        self.driver_limit = driver_limit
        self.active_allocations = {}
        self.cached_blocks = []
        self.next_handle = 1

    def allocate(self, size_bytes):
        current_used = sum(self.active_allocations.values()) + sum(self.cached_blocks)
        if current_used + size_bytes > self.driver_limit:
            raise RuntimeError("MPS backend out of memory")
        handle = self.next_handle
        self.next_handle += 1
        self.active_allocations[handle] = size_bytes
        return handle

    def free(self, handle):
        if handle in self.active_allocations:
            size = self.active_allocations.pop(handle)
            self.cached_blocks.append(size)

    def empty_cache(self):
        reclaimed = sum(self.cached_blocks)
        self.cached_blocks.clear()
        return reclaimed


def run_workload_with_defrag(allocator, trace):
    successful_allocs = 0
    recovered_ooms = 0
    total_reclaimed = 0

    for op, val in trace:
        if op == "alloc":
            try:
                handle = allocator.allocate(val)
                successful_allocs += 1
            except RuntimeError as e:
                if "MPS backend out of memory" in str(e):
                    reclaimed = allocator.empty_cache()
                    total_reclaimed += reclaimed
                    try:
                        handle = allocator.allocate(val)
                        successful_allocs += 1
                        recovered_ooms += 1
                    except RuntimeError:
                        pass
                else:
                    raise e
        elif op == "free":
            allocator.free(val)

    return {
        "successful_allocations": successful_allocs,
        "recovered_oom_count": recovered_ooms,
        "total_reclaimed_bytes": total_reclaimed,
    }
