class MockEngine:
    def __init__(self, kind):
        self.kind = kind
        self.runs = 0

    def generate(self, prompt_len, dtype):
        self.runs += 1
        if self.kind == "mlx":
            if self.runs == 1:
                return 500.0 + 2.0 * prompt_len
            return 10.0 + 2.0 * prompt_len
        elif self.kind == "llama":
            if self.runs == 1:
                return 50.0 + 3.0 * prompt_len
            return 5.0 + 3.0 * prompt_len
        return 0.0


def measure_warmup_bias(engine, prompt_len, warmup_runs, num_runs, dtype):
    latencies = []
    for _ in range(warmup_runs + num_runs):
        latencies.append(engine.generate(prompt_len, dtype))
    naive = sum(latencies) / len(latencies)
    steady = sum(latencies[warmup_runs:]) / num_runs
    return steady, naive - steady


def compare_engines(engine_mlx, engine_llama, prompt_lens, warmup_runs, num_runs, dtype):
    res = []
    for l in prompt_lens:
        mlx_s, _ = measure_warmup_bias(engine_mlx, l, warmup_runs, num_runs, dtype)
        llama_s, _ = measure_warmup_bias(engine_llama, l, warmup_runs, num_runs, dtype)
        res.append({
            "len": l,
            "mlx": mlx_s,
            "llama": llama_s,
            "mlx_slower": mlx_s > llama_s
        })
    return res
