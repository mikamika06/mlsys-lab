class KernelRun:
    def __init__(self, name: str, flops: int, bytes_accessed: int, time_ms: float):
        self.name = name
        self.flops = flops
        self.bytes_accessed = bytes_accessed
        self.time_ms = time_ms


class Hardware:
    def __init__(self, peak_gflops: float, peak_gbps: float):
        self.peak_gflops = peak_gflops
        self.peak_gbps = peak_gbps
