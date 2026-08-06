def compute_padding_waste(requests, max_batch_size):
    if not requests:
        return {
            "static_padded_tokens": 0,
            "static_useful_tokens": 0,
            "static_waste_ratio": 0.0,
            "continuous_padded_tokens": 0,
            "continuous_useful_tokens": 0,
            "continuous_waste_ratio": 0.0,
        }

    total_useful = sum(r["prompt_len"] + r["decode_len"] for r in requests)

    # Static Batching Calculation
    # Requests are chunked into fixed static batches of size up to max_batch_size.
    static_padded_total = 0
    for i in range(0, len(requests), max_batch_size):
        chunk = requests[i : i + max_batch_size]
        max_prompt = max(r["prompt_len"] for r in chunk)
        max_decode = max(r["decode_len"] for r in chunk)
        batch_padded = len(chunk) * (max_prompt + max_decode)
        static_padded_total += batch_padded

    static_useful = total_useful
    static_waste = static_padded_total - static_useful
    static_waste_ratio = float(static_waste) / float(static_padded_total) if static_padded_total > 0 else 0.0

    # Continuous Batching Calculation
    # Iteration-level continuous scheduling pads prompts within a step to max prompt in step (or 0 if prefill step isolated)
    # and decode steps require no padding across active requests since sequences run independently per slot.
    cont_padded_total = 0
    for r in requests:
        cont_padded_total += r["prompt_len"] + r["decode_len"]

    continuous_waste_ratio = 0.0

    return {
        "static_padded_tokens": static_padded_total - static_useful,
        "static_useful_tokens": static_useful,
        "static_waste_ratio": float(static_waste_ratio),
        "continuous_padded_tokens": 0,
        "continuous_useful_tokens": total_useful,
        "continuous_waste_ratio": continuous_waste_ratio,
    }
