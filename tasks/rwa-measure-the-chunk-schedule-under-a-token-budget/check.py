def _ref(prompt_len, max_tokens_per_step):
    if prompt_len <= 0:
        return 0, []
    steps = (prompt_len + max_tokens_per_step - 1) // max_tokens_per_step
    prefill_counts = [max_tokens_per_step] * (steps - 1)
    last = prompt_len - max_tokens_per_step * (steps - 1)
    prefill_counts.append(last)
    return steps, prefill_counts

def grade(sol, fx):
    cases = [
        (0, 4),
        (10, 4),
        (8, 4),
        (15, 5),
        (7, 3),
        (12, 6),
    ]
    ok = 1.0
    for prompt_len, budget in cases:
        try:
            got = sol.chunk_schedule(prompt_len, budget)
        except Exception:
            return {"exact_match": 0.0}
        ref = _ref(prompt_len, budget)
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
