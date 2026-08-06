class OutOfMemoryError(Exception):
    pass


class MPSDevice:
    def __init__(self, total_mem=16000):
        self._total = total_mem
        self._recommended = int(total_mem * 0.7)
        self.driver_allocs = []
        self.tensors = {}
        self.next_tid = 1
        self.next_did = 1

    def current_allocated_memory(self):
        return sum(sz for did, sz in self.tensors.values())

    def driver_allocated_memory(self):
        return sum(sz for did, sz, is_free in self.driver_allocs)

    def recommended_max_memory(self):
        return self._recommended

    def allocate(self, size):
        best_idx = -1
        for i, (did, dsz, is_free) in enumerate(self.driver_allocs):
            if is_free and dsz >= size:
                if best_idx == -1 or dsz < self.driver_allocs[best_idx][1]:
                    best_idx = i

        if best_idx != -1:
            did, dsz, _ = self.driver_allocs[best_idx]
            self.driver_allocs[best_idx] = (did, dsz, False)
            tid = self.next_tid
            self.next_tid += 1
            self.tensors[tid] = (did, size)
            return tid

        if self.driver_allocated_memory() + size > self._total:
            raise OutOfMemoryError("MPS backend out of memory")

        did = self.next_did
        self.next_did += 1
        self.driver_allocs.append((did, size, False))
        tid = self.next_tid
        self.next_tid += 1
        self.tensors[tid] = (did, size)
        return tid

    def free(self, tid):
        if tid in self.tensors:
            did, size = self.tensors.pop(tid)
            for i, (d, dsz, is_free) in enumerate(self.driver_allocs):
                if d == did:
                    self.driver_allocs[i] = (d, dsz, True)
                    break

    def empty_cache(self):
        self.driver_allocs = [d for d in self.driver_allocs if not d[2]]
