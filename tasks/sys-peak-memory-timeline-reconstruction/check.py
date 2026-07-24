def _oracle(events):
    live = 0
    peak = 0
    peak_step = 0
    for step, (kind, amount) in enumerate(events):
        if kind == "alloc":
            live += amount
        elif kind == "free":
            live -= amount
        if live > peak:
            peak = live
            peak_step = step
    return peak, peak_step


def grade(sol, fx) -> dict:
    cases = [
        [
            ("alloc", 400),
            ("alloc", 900),
            ("free", 300),
            ("alloc", 200),
        ],
        [
            ("alloc", 100),
            ("free", 50),
            ("alloc", 200),
            ("free", 100),
            ("free", 150),
        ],
        [
            ("alloc", 1024),
            ("alloc", 2048),
            ("free", 1024),
            ("alloc", 4096),
            ("free", 4096),
            ("free", 2048),
        ],
        [
            ("alloc", 7),
            ("alloc", 7),
            ("free", 7),
            ("free", 7),
        ],
        [
            ("alloc", 10),
            ("free", 10),
            ("alloc", 20),
            ("alloc", 30),
            ("free", 5),
        ],
    ]

    ok = 1.0
    for events in cases:
        try:
            got = tuple(sol.peak_memory_timeline(list(events)))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(events):
            ok = 0.0
            break
    return {"exact_match": ok}
