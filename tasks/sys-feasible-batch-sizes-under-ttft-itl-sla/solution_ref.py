def feasible_batch_sizes(
    batch_sizes,
    prompt_tokens,
    gen_tokens,
    sla_ttft_ms,
    sla_itl_ms,
):
    result = []
    for b in batch_sizes:
        ttft = 20.0 + 0.05 * prompt_tokens * b
        itl = 5.0 + 0.01 * gen_tokens * b
        result.append(ttft <= sla_ttft_ms and itl <= sla_itl_ms)
    return result
