def peak_memory_timeline(events):
    live = 0
    peak = 0
    peak_step = 0

    for step, (kind, amount) in enumerate(events):
        if kind == "alloc":
            live += amount
        else:
            live -= amount

        if live > peak:
            peak = live
            peak_step = step

    return peak, peak_step
