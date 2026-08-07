def find_wrong_token_type(tokens, token_types):
    for t, ty in zip(tokens, token_types):
        if t.startswith("<|") and t.endswith("|>") and ty != "control":
            return t
        if not t.startswith("<|") and ty == "control":
            return t
    return None
