import itertools


def _simulate(requests, kv_budget, order):
    waiting = list(order)
    active = []
    remaining = {}
    admitted = set()
    steps = 0

    while len(admitted) < len(requests) or active:
        used = sum(requests[i][2] * (requests[i][0] + requests[i][1]) for i in active)
        pos = 0
        while pos < len(waiting):
            i = waiting[pos]
            mem = requests[i][2] * (requests[i][0] + requests[i][1])
            if used + mem <= kv_budget:
                active.append(i)
                remaining[i] = requests[i][1]
                admitted.add(i)
                used += mem
                waiting.pop(pos)
            else:
                pos += 1

        if active:
            steps += 1
            done = []
            for i in active:
                remaining[i] -= 1
                if remaining[i] == 0:
                    done.append(i)
            for i in done:
                active.remove(i)
        elif waiting:
            # If no request can fit, the queue is infeasible.
            return 10**9

    return steps


def _oracle_opt(requests, kv_budget):
    best = 10**9
    for perm in itertools.permutations(range(len(requests))):
        best = min(best, _simulate(requests, kv_budget, perm))
    return best


def grade(sol, fx) -> dict:
    cases = [
        (
            [
                (8, 7, 1),
                (2, 3, 3),
                (4, 5, 1),
                (6, 2, 2),
            ],
            50,
        ),
        (
            [
                (10, 9, 1),
                (1, 2, 4),
                (3, 6, 1),
            ],
            45,
        ),
        (
            [
                (4, 4, 2),
                (7, 3, 1),
                (2, 9, 1),
                (5, 5, 1),
            ],
            45,
        ),
    ]

    worst = 1.0
    for requests, budget in cases:
        optimum = _oracle_opt(requests, budget)
        try:
            order = list(sol.schedule_queue(requests, budget))
            candidate = _simulate(requests, budget, order)
        except Exception:
            return {"size_ratio": 10**9}

        if sorted(order) != list(range(len(requests))):
            return {"size_ratio": 10**9}

        worst = max(worst, candidate / optimum)

    return {"size_ratio": float(worst)}
