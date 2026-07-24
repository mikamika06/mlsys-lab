def _reference(events):
    B = sum(1 for e in events if e.startswith("break_"))
    S = 0
    in_block = False
    for e in events:
        if e.startswith("break_"):
            in_block = False
        else:
            if not in_block:
                S += 1
                in_block = True
    return B, S

def grade(sol, fx) -> dict:
    cases = [
        [],
        ["op"],
        ["break_a"],
        ["op", "op2"],
        ["break_a", "op", "break_b", "op2", "op3"],
        ["break_a","break_b","break_c"],
        ["op1","break_a","op2","op3","break_b","op4"],
    ]
    for events in cases:
        try:
            got = sol.count_breaks_and_subgraphs(events)
        except Exception:
            return {"exact_match": 0.0}
        expected = _reference(events)
        if got != expected:
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}
