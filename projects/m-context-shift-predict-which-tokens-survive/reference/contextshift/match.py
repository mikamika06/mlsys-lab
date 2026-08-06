def find_surviving_tokens(old_tokens, new_tokens):
    """Find indices of tokens that survive a context shift."""
    surviving = []
    for i, (o, n) in enumerate(zip(old_tokens, new_tokens)):
        if o == n:
            surviving.append(i)
        else:
            break
    return surviving
