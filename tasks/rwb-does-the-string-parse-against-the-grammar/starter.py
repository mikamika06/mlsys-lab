def run_fsm(transitions: dict, start_state, accept_states: set, s: str) -> bool:
    """
    Run a deterministic finite-state machine over `s` and report whether it
    is ACCEPTED. transitions maps (state, char) -> next_state; a missing
    entry means the machine is stuck (dead state) and the string is
    rejected. After consuming `s`, accept iff the final state is in
    `accept_states`.
    """
    raise NotImplementedError('your code here')
