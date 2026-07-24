from collections import defaultdict


def allowed_next_tokens(transitions: dict, current_state, accept_states: set, vocab: list) -> set:
    """
    Grammar-constrained-decoding style DFA query: which vocabulary tokens
    (characters) are SAFE to emit next from `current_state`, in the sense
    that doing so keeps a path to eventual acceptance alive.

    transitions   : dict (state, char) -> next_state -- the COMPLETE DFA.
    current_state : the state the decoder is currently in.
    accept_states : set of accepting states.
    vocab         : candidate next tokens to test.

    A token c is allowed iff:
      1. (current_state, c) is a defined transition, AND
      2. transitions[(current_state, c)] can still reach SOME state in
         accept_states via zero or more further transitions (it is not a
         dead/trap state).

    Condition 2 requires computing, for every state in the machine,
    whether an accept state is reachable from it -- this is found by a
    breadth-first search over the REVERSED transition graph, starting
    from `accept_states` (a state can reach an accept state iff the
    accept state can be reached FROM it walking edges backwards).
    """
    # reverse adjacency: s2 -> {s : (s, c) -> s2 for some c}
    rev = defaultdict(set)
    for (s, _c), s2 in transitions.items():
        rev[s2].add(s)

    alive = set(accept_states)
    frontier = list(accept_states)
    while frontier:
        s2 = frontier.pop()
        for s in rev[s2]:
            if s not in alive:
                alive.add(s)
                frontier.append(s)

    return {c for c in vocab if (current_state, c) in transitions and transitions[(current_state, c)] in alive}
