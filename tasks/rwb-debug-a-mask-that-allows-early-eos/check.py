def _oracle(states, transitions, accepting, eos):
    result = {}
    for state in states:
        allowed = set()
        outgoing = transitions.get(state, {})
        for token in outgoing:
            allowed.add(token)
        if state in accepting:
            allowed.add(eos)
        result[state] = allowed
    return result


def grade(sol, fx) -> dict:
    cases = [
        (
            ["s0", "s1", "s2"],
            {
                "s0": {"a": "s1"},
                "s1": {"b": "s2"},
                "s2": {}
            },
            {"s2"},
            "<EOS>",
        ),
        (
            ["root", "name", "comma", "end"],
            {
                "root": {"<": "name"},
                "name": {">": "comma"},
                "comma": {",": "end"},
                "end": {}
            },
            {"end"},
            0,
        ),
        (
            ["q0", "q1"],
            {
                "q0": {"x": "q1", "y": "q0"},
                "q1": {"z": "q1"}
            },
            {"q1"},
            "EOS",
        ),
        (
            ["a", "b", "c"],
            {
                "a": {},
                "b": {"1": "c"},
                "c": {"2": "a"}
            },
            {"a", "c"},
            None,
        ),
    ]

    for states, transitions, accepting, eos in cases:
        try:
            got = sol.allowed_tokens(
                list(states),
                {k: dict(v) for k, v in transitions.items()},
                set(accepting),
                eos,
            )
        except Exception:
            return {"exact_match": 0.0}

        expected = _oracle(states, transitions, accepting, eos)
        if got != expected:
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}
