def explain_zero_eval_count(prompt_len: int, metrics: dict) -> str:
    if prompt_len > 0 and metrics.get("prompt_eval_count", 0) == 0:
        return "Prefill phase was bypassed or cached entirely, or metrics counter was reset before recording."
    return "Normal evaluation count."


def compare_runners(runner_a: dict, runner_b: dict) -> str:
    throughput_a = runner_a.get("total_tokens", 0) / max(runner_a.get("elapsed_sec", 1e-6), 1e-6)
    throughput_b = runner_b.get("total_tokens", 0) / max(runner_b.get("elapsed_sec", 1e-6), 1e-6)
    if throughput_a >= throughput_b:
        return "runner_a"
    return "runner_b"
