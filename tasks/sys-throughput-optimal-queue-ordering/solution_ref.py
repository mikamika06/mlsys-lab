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
            return 10**9

    return steps


def schedule_queue(requests, kv_budget):
    best_order = None
    best_cost = 10**9
    for order in itertools.permutations(range(len(requests))):
        cost = _simulate(requests, kv_budget, order)
        if cost < best_cost:
            best_cost = cost
            best_order = order
    return list(best_order)
