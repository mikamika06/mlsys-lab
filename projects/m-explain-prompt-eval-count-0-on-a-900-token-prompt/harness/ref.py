SCENARIOS = [
    {
        "prompt_len": 900,
        "metrics": {"prompt_eval_count": 0},
        "runner_a": {"total_tokens": 1000, "elapsed_sec": 2.0},
        "runner_b": {"total_tokens": 800, "elapsed_sec": 2.0},
        "context_len": 4000,
        "params": {"time_per_prefill_token": 0.001, "fixed_overhead": 0.02}
    }
]
