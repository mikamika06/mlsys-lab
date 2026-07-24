from collections import defaultdict


def _reachable_to_accept(transitions, accept_states):
    """All states that can reach SOME accept state via 0+ transitions,
    computed by BFS over the REVERSED transition graph starting from the
    accept states."""
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
    return alive


def _oracle_allowed(transitions, alive, state, vocab):
    return {c for c in vocab if (state, c) in transitions and transitions[(state, c)] in alive}


# Grammar 1: a*b  (zero or more 'a', then exactly one 'b', then nothing)
_G1_TRANSITIONS = {
    ('q0', 'a'): 'q0', ('q0', 'b'): 'q1', ('q0', 'c'): 'qtrap',
    ('q1', 'a'): 'qtrap', ('q1', 'b'): 'qtrap', ('q1', 'c'): 'qtrap',
    ('qtrap', 'a'): 'qtrap', ('qtrap', 'b'): 'qtrap', ('qtrap', 'c'): 'qtrap',
}
_G1_START = 'q0'
_G1_ACCEPT = {'q1'}
_G1_PREFIX = "aab"

# Grammar 2: (ab)*c  (zero or more "ab" pairs, then exactly one 'c')
_G2_TRANSITIONS = {
    ('r0', 'a'): 'r1', ('r0', 'b'): 'trap', ('r0', 'c'): 'r2',
    ('r1', 'a'): 'trap', ('r1', 'b'): 'r0', ('r1', 'c'): 'trap',
    ('r2', 'a'): 'trap', ('r2', 'b'): 'trap', ('r2', 'c'): 'trap',
    ('trap', 'a'): 'trap', ('trap', 'b'): 'trap', ('trap', 'c'): 'trap',
}
_G2_START = 'r0'
_G2_ACCEPT = {'r2'}
_G2_PREFIX = "abab"

_VOCAB = ['a', 'b', 'c']


def _walk_states(transitions, start, prefix):
    state = start
    states = [state]
    for ch in prefix:
        state = transitions[(state, ch)]
        states.append(state)
    return states


def grade(sol, fx) -> dict:
    ok = 1.0

    for transitions, start, accept, prefix in (
        (_G1_TRANSITIONS, _G1_START, _G1_ACCEPT, _G1_PREFIX),
        (_G2_TRANSITIONS, _G2_START, _G2_ACCEPT, _G2_PREFIX),
    ):
        alive = _reachable_to_accept(transitions, accept)
        states_visited = _walk_states(transitions, start, prefix)

        for state in states_visited:
            ref = _oracle_allowed(transitions, alive, state, _VOCAB)
            try:
                got = sol.allowed_next_tokens(dict(transitions), state, set(accept), list(_VOCAB))
                got = set(got)
            except Exception:
                return {"exact_match": 0.0}
            if got != ref:
                ok = 0.0

    return {"exact_match": ok}
