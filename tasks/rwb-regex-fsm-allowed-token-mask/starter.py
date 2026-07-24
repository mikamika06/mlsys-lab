def allowed_next_tokens(transitions: dict, current_state, accept_states: set, vocab: list) -> set:
    """
    Return the set of tokens c in `vocab` such that (current_state, c) is a
    defined transition in `transitions` AND the resulting state can still
    reach some state in `accept_states` (it is not a dead/trap state).
    See task.md.
    """
    raise NotImplementedError('your code here')
