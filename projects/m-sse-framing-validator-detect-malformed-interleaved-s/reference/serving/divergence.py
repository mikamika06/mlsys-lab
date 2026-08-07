def locate_template_divergence(tokens_a, tokens_b):
    if not isinstance(tokens_a, list) or not isinstance(tokens_b, list):
        return 0
    for i, (a, b) in enumerate(zip(tokens_a, tokens_b)):
        if a != b:
            return i
    return min(len(tokens_a), len(tokens_b))
