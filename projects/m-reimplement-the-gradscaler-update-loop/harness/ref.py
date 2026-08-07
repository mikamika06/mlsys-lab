import math

CONFIGS = [
    {
        "history": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, float("inf")], [1.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]]],
        "initial": 1024.0,
        "interval": 2,
    },
    {
        "history": [[[0.1, 0.2]] for _ in range(5)],
        "initial": 2048.0,
        "growth_interval": 3,
    },
    {
        "history": [[[float("nan"), 1.0]], [[1.0, 1.0]]],
        "initial": 512.0,
        "growth_interval": 10,
    },
]


def find_safe_scale(grads, current_scale, growth_factor=2.0, backoff_factor=0.5):
    has_inf = False
    for g in grads:
        for val in g:
            if not math.isfinite(val):
                has_inf = True
                break
        if has_inf:
            break
    if has_inf:
        return current_scale * backoff_factor, True
    return current_scale, False


def update_scaler(state, has_inf):
    if has_inf:
        state["scale"] = state["scale"] * state["backoff_factor"]
        state["growth_track"] = 0
    else:
        state["growth_track"] += 1
        if state["growth_track"] >= state["growth_interval"]:
            state["scale"] = state["scale"] * state["growth_factor"]
            state["growth_track"] = 0
    return state["scale"]


def simulate_trajectory(grad_history, initial_scale=65536.0, growth_interval=2000, growth_factor=2.0, backoff_factor=0.5):
    state = {
        "scale": initial_scale,
        "growth_track": 0,
        "growth_interval": growth_interval,
        "growth_factor": growth_factor,
        "backoff_factor": backoff_factor,
    }
    scales = []
    for grads in grad_history:
        _, has_inf = find_safe_scale(grads, state["scale"])
        current = update_scaler(state, has_inf)
        scales.append(current)
    return scales
