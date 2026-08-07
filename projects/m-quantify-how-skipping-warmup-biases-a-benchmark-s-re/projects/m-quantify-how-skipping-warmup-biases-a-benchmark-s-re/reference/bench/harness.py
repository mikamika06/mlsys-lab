import numpy as np

class BenchmarkHarness:
    def __init__(self, prompt_len, seq_len, dtype="float16", backend="mlx"):
        self.prompt_len = prompt_len
        self.seq_len = seq_len
        self.dtype = dtype
        self.backend = backend

    def run(self, warmup=5, iters=20):
        rng = np.random.default_rng(42 + self.prompt_len + self.seq_len)
        base = 0.01 + 0.0002 * self.seq_len
        if self.backend == "mlx":
            base *= 1.2
        elif self.backend == "llama_cpp":
            base *= 0.85
        lats = []
        for i in range(iters):
            overhead = 0.08 * np.exp(-i / 1.5) if i < warmup else 0.0
            noise = rng.normal(0, 0.001)
            lats.append(max(0.001, base + overhead + noise))
        return lats
