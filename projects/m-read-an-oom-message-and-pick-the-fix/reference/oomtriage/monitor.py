class MockCudaMemory:
    def __init__(self):
        self.allocated = 0
        self.reserved = 0
        self.max_allocated = 0
        self.max_reserved = 0

    def alloc(self, size):
        self.allocated += size
        self.reserved += size
        self.max_allocated = max(self.max_allocated, self.allocated)
        self.max_reserved = max(self.max_reserved, self.reserved)

    def free(self, size):
        self.allocated -= size
        self.reserved = max(self.reserved, self.allocated)


def track_loop_memory(loop_func, *args, **kwargs) -> dict:
    alloc_history = []
    reserved_history = []
    mock_cuda = MockCudaMemory()
    def step_callback(alloc_sz, free_sz):
        mock_cuda.alloc(alloc_sz)
        alloc_history.append(mock_cuda.allocated)
        reserved_history.append(mock_cuda.reserved)
        if free_sz > 0:
            mock_cuda.free(free_sz)
            alloc_history.append(mock_cuda.allocated)
            reserved_history.append(mock_cuda.reserved)
    loop_func(step_callback, *args, **kwargs)
    return {
        "peak_allocated": mock_cuda.max_allocated,
        "peak_reserved": mock_cuda.max_reserved,
        "allocated_history": alloc_history,
        "reserved_history": reserved_history,
    }
