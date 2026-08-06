def measure_throughput(prompts, max_new_tokens):
    """Calculates operation counts and speedup ratio between pipeline and handrolled generation."""
    total_tokens = len(prompts) * max_new_tokens
    handrolled_ops = 0
    pipeline_ops = 0
    for prompt in prompts:
        prompt_len = len(prompt)
        for i in range(max_new_tokens):
            handrolled_ops += prompt_len + i
            pipeline_ops += 1
    speedup = float(handrolled_ops) / float(max(1, pipeline_ops))
    return {
        "total_generated_tokens": float(total_tokens),
        "handrolled_ops": float(handrolled_ops),
        "pipeline_ops": float(pipeline_ops),
        "speedup_ratio": speedup,
    }
