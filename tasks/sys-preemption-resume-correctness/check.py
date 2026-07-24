def _next_token(state):
    state = (1103515245 * state + 12345) % (2 ** 31)
    return state, state % 1000


def _oracle(requests):
    result = {}
    for req in requests:
        state = req["seed"]
        tokens = []
        for _ in range(req["steps"]):
            state, token = _next_token(state)
            tokens.append(token)
        result[req["id"]] = tokens
    return result


def grade(sol, fx) -> dict:
    cases = [
        [
            {"id": 1, "seed": 3, "steps": 5},
            {"id": 2, "seed": 8, "steps": 4},
        ],
        [
            {"id": 10, "seed": 123, "steps": 1},
            {"id": 11, "seed": 456, "steps": 7},
            {"id": 12, "seed": 789, "steps": 3},
        ],
        [
            {"id": 5, "seed": 0, "steps": 9},
            {"id": 6, "seed": 99, "steps": 2},
        ],
    ]
    ok = 1.0
    for requests in cases:
        for quantum in [1, 2, 4]:
            try:
                got = sol.resume_decode(
                    [dict(r) for r in requests],
                    quantum,
                )
            except Exception:
                ok = 0.0
                break
            if got != _oracle(requests):
                ok = 0.0
                break
        if ok == 0.0:
            break
    return {"exact_match": ok}
