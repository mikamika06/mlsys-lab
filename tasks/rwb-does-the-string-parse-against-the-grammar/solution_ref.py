def run_fsm(transitions: dict, start_state, accept_states: set, s: str) -> bool:
    """
    Run a deterministic finite-state machine (the standard execution model
    behind regex engines and grammar-constrained LLM decoding) over `s` and
    report whether it is ACCEPTED.

    transitions   : dict mapping (state, char) -> next_state.
    start_state   : the FSM's initial state.
    accept_states : set/frozenset of accepting states.
    s             : candidate string.

    Walk the machine one character at a time starting from `start_state`.
    If, at any point, (state, char) has no entry in `transitions`, the
    machine is STUCK (an implicit dead state with no outgoing transitions)
    and the string is rejected immediately. After the whole string has been
    consumed, the string is accepted iff the final state is in
    `accept_states` (the empty string is accepted iff `start_state` itself
    is accepting).
    """
    state = start_state
    for ch in s:
        key = (state, ch)
        if key not in transitions:
            return False
        state = transitions[key]
    return state in accept_states
