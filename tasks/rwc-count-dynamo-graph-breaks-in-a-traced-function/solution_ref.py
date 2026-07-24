def count_breaks_and_subgraphs(events):
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
