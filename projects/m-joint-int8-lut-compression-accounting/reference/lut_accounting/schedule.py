def allocation_schedule(config, alignment, steps):
    from .memory import calculate_bytes
    base = calculate_bytes(config, alignment)
    schedule = []
    for step in range(steps):
        factor = max(0.0, 1.0 - (step / max(1, steps - 1)))
        current = int(base * factor)
        schedule.append(current)
    return schedule
