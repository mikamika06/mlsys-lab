def derive_max_num_batched_tokens(waiting_prompts, running_decodes, target_prefill_ratio):
    total_waiting_tokens = sum(p["prompt_tokens"] for p in waiting_prompts)
    total_running_tokens = sum(r["seq_len"] for r in running_decodes)
    if not waiting_prompts:
        return total_running_tokens
    avg_waiting_size = total_waiting_tokens / len(waiting_prompts)
    recommended = int(total_running_tokens * (1.0 - target_prefill_ratio) + avg_waiting_size * target_prefill_ratio)
    return max(128, recommended)
