from scaler.core import find_safe_scale, update_scaler


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
