def num_predict_budget(num_predict, prompt_tokens, context_size, hard_cap):
    remaining = max(context_size - prompt_tokens, 0)
    if num_predict >= 0:
        return min(num_predict, remaining)
    if num_predict == -2:
        return remaining
    if num_predict == -1:
        return max(hard_cap - prompt_tokens, 0)
    raise ValueError(f"unsupported num_predict {num_predict}")
