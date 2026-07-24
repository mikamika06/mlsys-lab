def _oracle(requests, budget):
    state = [
        {
            "id": int(r["id"]),
            "kv": int(r["kv"]),
            "tokens": int(r["tokens"]),
        }
        for r in requests
    ]
    answer = []
    while any(r["tokens"] > 0 for r in state):
        running = []
        used = 0
        for r in state:
            if r["tokens"] > 0 and used + r["kv"] <= budget:
                running.append(r)
                used += r["kv"]
        step = []
        for r in running:
            step.append(r["id"])
            r["tokens"] -= 1
        answer.append(step)
    return answer


def grade(sol, fx) -> dict:
    cases = [
        (
            [
                {"id": 1, "kv": 4, "tokens": 2},
                {"id": 2, "kv": 3, "tokens": 1},
                {"id": 3, "kv": 5, "tokens": 1},
            ],
            7,
        ),
        (
            [
                {"id": 8, "kv": 2, "tokens": 3},
                {"id": 4, "kv": 2, "tokens": 2},
                {"id": 9, "kv": 4, "tokens": 1},
                {"id": 6, "kv": 3, "tokens": 2},
            ],
            6,
        ),
        (
            [
                {"id": 0, "kv": 10, "tokens": 2},
                {"id": 1, "kv": 1, "tokens": 1},
                {"id": 2, "kv": 1, "tokens": 2},
            ],
            10,
        ),
        (
            [
                {"id": 5, "kv": 3, "tokens": 4},
                {"id": 7, "kv": 3, "tokens": 4},
                {"id": 11, "kv": 3, "tokens": 1},
            ],
            6,
        ),
    ]
    ok = 1.0
    for requests, budget in cases:
        try:
            got = sol.admit_requests(
                [dict(r) for r in requests],
                budget,
            )
        except Exception:
            ok = 0.0
            break
        if got != _oracle(requests, budget):
            ok = 0.0
            break
    return {"exact_match": ok}
