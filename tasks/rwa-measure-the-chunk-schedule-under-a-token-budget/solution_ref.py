def chunk_schedule(prompt_len, max_tokens_per_step):
    """
    Return the number of steps and a list with the token count for each step.
    Handles non‑negative integer inputs; zero prompt length yields (0, []).
    """
    if prompt_len <= 0:
        return 0, []
    steps = (prompt_len + max_tokens_per_step - 1) // max_tokens_per_step
    prefill_counts = [max_tokens_per_step] * (steps - 1)
    last = prompt_len - max_tokens_per_step * (steps - 1)
    prefill_counts.append(last)
    return steps, prefill_counts
