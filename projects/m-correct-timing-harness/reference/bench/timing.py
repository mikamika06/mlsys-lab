"""CUDA-aware timing harness module."""
import time
import numpy as np


class MockCudaEvent:
    """Mock CUDA Event for deterministic timing harness simulation."""
    def __init__(self):
        self.recorded_time = 0.0

    def record(self, stream=None):
        self.recorded_time = time.perf_counter()

    def synchronize(self):
        pass

    def elapsed_time(self, end_event):
        return (end_event.recorded_time - self.recorded_time) * 1000.0


def flush_l2_cache(l2_size_mb=40):
    """Flushes L2 cache by allocating and accessing a zeroed array larger than L2 size."""
    size_bytes = l2_size_mb * 1024 * 1024
    num_elements = size_bytes // 4
    arr = np.zeros(num_elements, dtype=np.float32)
    _ = np.sum(arr)


def measure_kernel_execution(fn, warmup_iters=10, active_iters=50, flush_l2=True, l2_size_mb=40):
    """Executes a CUDA function with proper synchronization, warmup, and L2 cache flushing."""
    for _ in range(warmup_iters):
        if flush_l2:
            flush_l2_cache(l2_size_mb)
        fn()

    times = []
    for _ in range(active_iters):
        if flush_l2:
            flush_l2_cache(l2_size_mb)

        start_evt = MockCudaEvent()
        end_evt = MockCudaEvent()

        start_time = time.perf_counter()
        start_evt.record()
        fn()
        end_evt.record()
        end_evt.synchronize()
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000.0
        times.append(elapsed_ms)

    return np.array(times, dtype=np.float64)


def synchronize_and_time(fn, stream=None):
    """Measures precise execution time of a single invocation with explicit stream synchronization."""
    start = time.perf_counter()
    fn()
    end = time.perf_counter()
    return (end - start) * 1000.0
