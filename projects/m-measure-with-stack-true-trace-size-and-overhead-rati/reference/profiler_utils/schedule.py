def active_windows_at_step(step: int, wait: int, warmup: int, active: int, repeat: int) -> int:
    if step < 0:
        return 0
    cycle_len = wait + warmup + active
    if cycle_len == 0:
        return 0
    count = 0
    for s in range(step + 1):
        c = s // cycle_len
        if repeat > 0 and c >= repeat:
            break
        pos = s % cycle_len
        if pos >= wait + warmup:
            count += 1
    return count
