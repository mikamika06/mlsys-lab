def predict_token_counts(prompts, completions):
    res = []
    for p, c in zip(prompts, completions):
        p_tokens = len(p.split())
        c_tokens = len(c.split())
        res.append({"prompt_tokens": p_tokens, "completion_tokens": c_tokens, "total_tokens": p_tokens + c_tokens})
    return res
