def _reference(ops):
    events = 0
    owned = True
    for op in ops:
        if op == "io":
            if owned:
                events += 1
                owned = False
            if not owned:
                events += 1
                owned = True
        elif op == "alloc":
            if owned:
                events += 1
                owned = False
            if not owned:
                events += 1
                owned = True
        elif op == "compute":
            pass
    return events


def grade(sol, fx) -> dict:
    cases = [
        ["compute", "compute"],
        ["io"],
        ["alloc", "compute", "io"],
        ["compute", "alloc", "alloc", "compute"],
        ["io", "io", "alloc", "compute", "io"],
        ["compute"] * 20 + ["io", "alloc"],
    ]
    ok = 1.0
    for ops in cases:
        try:
            got = sol.modeled_gil_count(list(ops))
        except Exception:
            ok = 0.0
            break
        if got != _reference(ops):
            ok = 0.0
            break
    return {"exact_match": ok}
