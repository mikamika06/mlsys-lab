def allowed_tokens(states, transitions, accepting, eos):
    # TODO: this leaves EOS enabled for every state, which allows decoding to
    # terminate before the FSM has reached an accepting state.
    masks = {}
    for state in states:
        tokens = set(transitions.get(state, {}).keys())
        tokens.add(eos)
        masks[state] = tokens
    return masks
