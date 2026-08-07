def compute_penalty(cached_tokens, requested_tokens, compute_cost_per_token):
    matched = 0
    for c, r in zip(cached_tokens, requested_tokens):
        if c == r:
            matched += 1
        else:
            break
    missed = len(requested_tokens) - matched
    return float(missed * compute_cost_per_token)
