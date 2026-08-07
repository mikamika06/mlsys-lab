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
