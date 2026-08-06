from mpsmem.tracker import MPSMemoryTracker


class SimulatedMPSAllocator:
    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = capacity_bytes
        self.tracker = MPSMemoryTracker(capacity_bytes)

    def allocate(self, block_id: str, size_bytes: int) -> bool:
        current_driver = self.tracker.driver_allocated_memory()
        if current_driver + size_bytes > self.capacity_bytes:
            return False
        self.tracker.allocate(block_id, size_bytes)
        return True

    def free(self, block_id: str) -> None:
        self.tracker.free(block_id)

    def empty_cache(self) -> int:
        return self.tracker.empty_cache()

    def run_workload(self, operations: list) -> dict:
        oom_occurred = False
        ops_completed = 0
        for op in operations:
            kind = op[0]
            if kind == "alloc":
                bid, size = op[1], op[2]
                success = self.allocate(bid, size)
                if not success:
                    oom_occurred = True
                    break
            elif kind == "free":
                bid = op[1]
                self.free(bid)
            elif kind == "empty_cache":
                self.empty_cache()
            ops_completed += 1

        return {
            "completed": ops_completed,
            "oom": oom_occurred,
            "current_allocated": self.tracker.current_allocated_memory(),
            "driver_allocated": self.tracker.driver_allocated_memory(),
            "divergence": self.tracker.divergence_bytes(),
            "fragmentation_ratio": self.tracker.fragmentation_ratio(),
        }
