class SimulatedMPSAllocator:
    def __init__(self, driver_limit):
        raise NotImplementedError

    def allocate(self, size_bytes):
        raise NotImplementedError

    def free(self, handle):
        raise NotImplementedError

    def empty_cache(self):
        raise NotImplementedError


def run_workload_with_defrag(allocator, trace):
    """
    Executes allocation trace. On OOM, invokes empty_cache() and retries once.
    Returns (successful_allocations, recovered_oom_count, total_reclaimed_bytes).
    """
    raise NotImplementedError
