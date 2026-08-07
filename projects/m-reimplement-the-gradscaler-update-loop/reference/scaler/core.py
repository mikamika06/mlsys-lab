import math


def find_safe_scale(grads, current_scale, growth_factor=2.0, backoff_factor=0.5):
    has_inf = False
    max_val = 0.0
    for g in grads:
        for val in g:
            if not math.isfinite(val):
                has_inf = True
                break
            if abs(val) > max_val:
                max_val = abs(val)
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
