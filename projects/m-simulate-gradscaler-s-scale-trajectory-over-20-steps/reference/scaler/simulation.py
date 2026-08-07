def simulate_trajectory(overflows, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=5):
    scales = []
    current_scale = init_scale
    success_count = 0
    for ov in overflows:
        scales.append(current_scale)
        if ov:
            current_scale *= backoff_factor
            success_count = 0
        else:
            success_count += 1
            if success_count >= growth_interval:
                current_scale *= growth_factor
                success_count = 0
    return scales

def identify_overflows(overflows):
    return [i for i, ov in enumerate(overflows) if ov]

def next_doubling_step(overflows, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=5):
    scales = simulate_trajectory(overflows, init_scale, growth_factor, backoff_factor, growth_interval)
    target = init_scale * growth_factor
    for i, s in enumerate(scales):
        if s >= target:
            return i
    return -1
