import numpy as np

class BenchmarkHarness:
    def __init__(self, prompt_len, seq_len, dtype="float16", backend="mlx"):
        self.prompt_len = prompt_len
        self.seq_len = seq_len
        self.dtype = dtype
        self.backend = backend

    def run(self, warmup=5, iters=20):
        rng = np.random.default_rng(42 + self.prompt_len)
        base = 0.02 + 0.0001 * self.seq_len
        if self.backend == "mlx":
            base *= 1.15
        elif self.backend == "llama_cpp":
            base *= 0.90
        latencies = []
        for i in range(iters):
            overhead = 0.05 * np.exp(-i / 2.0) if i < warmup else 0.0
            noise = rng.normal(0, 0.002)
            latencies.append(max(0.005, base + overhead + noise))
        return latencies
