def simulate_trajectory(overflows, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
    scales = []
    current_scale = float(init_scale)
    successes = 0

    for is_overflow in overflows:
        scales.append(current_scale)
        if is_overflow:
            current_scale *= backoff_factor
            successes = 0
        else:
            successes += 1
            if successes == growth_interval:
                current_scale *= growth_factor
                successes = 0

    return scales


def identify_skipped_steps(overflows):
    return [i for i, ov in enumerate(overflows) if ov]


def next_doubling_step(overflows, growth_interval=2000):
    successes = 0
    for is_overflow in overflows:
        if is_overflow:
            successes = 0
        else:
            successes += 1
            if successes == growth_interval:
                successes = 0

    needed = growth_interval - successes
    return len(overflows) + needed - 1
