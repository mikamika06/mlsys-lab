def _next_token(state):
    state = (1103515245 * state + 12345) % (2 ** 31)
    return state, state % 1000


def resume_decode(requests, quantum):
    states = {}
    remaining = {}
    output = {}

    for req in requests:
        states[req["id"]] = req["seed"]
        remaining[req["id"]] = req["steps"]
        output[req["id"]] = []

    active = [req["id"] for req in requests]

    while active:
        next_active = []
        for rid in active:
            state = states[rid]
            count = 0
            while count < quantum and remaining[rid] > 0:
                state, token = _next_token(state)
                output[rid].append(token)
                remaining[rid] -= 1
                count += 1
            states[rid] = state
            if remaining[rid] > 0:
                next_active.append(rid)
        active = next_active

    return output
