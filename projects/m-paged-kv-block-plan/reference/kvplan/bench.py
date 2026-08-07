def generate_throughput_report(requests_completed, total_prompt_tokens, total_gen_tokens, total_time_sec):
    if total_time_sec <= 0:
        return {
            "token_throughput": 0.0,
            "prompt_throughput": 0.0,
            "generation_throughput": 0.0,
            "throughput_ratio": 0.0
        }

    total_tokens = total_prompt_tokens + total_gen_tokens
    token_throughput = total_tokens / total_time_sec
    prompt_throughput = total_prompt_tokens / total_time_sec
    gen_throughput = total_gen_tokens / total_time_sec

    baseline_throughput = 100.0
    throughput_ratio = token_throughput / baseline_throughput

    return {
        "requests_completed": requests_completed,
        "total_tokens": total_tokens,
        "token_throughput": token_throughput,
        "prompt_throughput": prompt_throughput,
        "generation_throughput": gen_throughput,
        "throughput_ratio": throughput_ratio
    }
