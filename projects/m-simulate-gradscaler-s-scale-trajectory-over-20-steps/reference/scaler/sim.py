def simulate_trajectory(overflows, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
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
            if success_count == growth_interval:
                current_scale *= growth_factor
                success_count = 0
    return scales


def identify_overflows(trajectory, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
    overflows = []
    current_scale = init_scale
    success_count = 0
    for s in trajectory:
        if abs(s - current_scale) > 1e-5:
            if s < current_scale:
                overflows.append(True)
                current_scale = s
                success_count = 0
            else:
                overflows.append(False)
                current_scale = s
                success_count = 0
        else:
            overflows.append(False)
            success_count += 1
            if success_count == growth_interval:
                current_scale *= growth_factor
                success_count = 0
    return overflows


def next_doubling_step(overflows, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
    current_scale = init_scale
    success_count = 0
    for ov in overflows:
        if ov:
            current_scale *= backoff_factor
            success_count = 0
        else:
            success_count += 1
            if success_count == growth_interval:
                current_scale *= growth_factor
                success_count = 0
    target_scale = current_scale * 2.0
    steps = 0
    while current_scale < target_scale and steps < 100000:
        success_count += 1
        steps += 1
        if success_count == growth_interval:
            current_scale *= growth_factor
            success_count = 0
    return steps
