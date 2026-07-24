import math


def _oracle(values):
    current = list(values)
    trace = []
    while len(current) > 1:
        nxt = []
        i = 0
        while i + 1 < len(current):
            nxt.append(current[i] + current[i + 1])
            i += 2
        if i < len(current):
            nxt.append(current[i])
        trace.append([len(nxt)])
        current = nxt
    return (current[0] if current else 0.0), trace


def grade(sol, fx) -> dict:
    cases = [
        [1.0],
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0.5, -1.25, 3.75, 2.0, 8.0, -4.0, 6.0],
        list(range(17)),
    ]

    ok = 1.0
    for values in cases:
        try:
            got_value, got_trace = sol.tree_reduce(list(values))
        except Exception:
            ok = 0.0
            break

        ref_value, _ = _oracle(values)
        expected_depth = math.ceil(math.log2(len(values))) if len(values) > 1 else 0

        if abs(float(got_value) - float(ref_value)) >= 1e-9:
            ok = 0.0
            break
        if len(got_trace) != expected_depth:
            ok = 0.0
            break

        for level in got_trace:
            if not isinstance(level, list) or len(level) != 1:
                ok = 0.0
                break
        if ok == 0.0:
            break

    return {"exact_match": ok}
