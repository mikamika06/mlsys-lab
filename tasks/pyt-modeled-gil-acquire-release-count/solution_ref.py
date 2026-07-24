def modeled_gil_count(ops):
    events = 0
    for op in ops:
        if op == "io" or op == "alloc":
            events += 2
    return events
