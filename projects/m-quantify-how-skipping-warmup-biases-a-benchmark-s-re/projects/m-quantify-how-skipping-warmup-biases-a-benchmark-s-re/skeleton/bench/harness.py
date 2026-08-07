class BenchmarkHarness:
    def __init__(self, prompt_len, seq_len, dtype="float16", backend="mlx"):
        raise NotImplementedError

    def run(self, warmup=5, iters=20):
        raise NotImplementedError
