def allowed_tokens(states, transitions, accepting, eos):
    masks = {}
    for state in states:
        tokens = set(transitions.get(state, {}).keys())
        if state in accepting:
            tokens.add(eos)
        masks[state] = tokens
    return masks
