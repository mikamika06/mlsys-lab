from contextshift.match import find_surviving_tokens

def survival_ratio(old_tokens, new_tokens):
    """Compute ratio of surviving tokens."""
    surviving = find_surviving_tokens(old_tokens, new_tokens)
    if not new_tokens:
        return 0.0
    return float(len(surviving)) / float(len(new_tokens))
