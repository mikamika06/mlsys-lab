def _oracle(stages, microbatches):
    def deps(op):
        s, m, phase = op
        if phase == "F":
            return [] if s == 0 else [(s - 1, m, "F")]
        return [(s, m, "F")] + ([] if s == stages - 1 else [(s + 1, m, "B")])

    remaining = set()
    for s in range(stages):
        for m in range(microbatches):
            remaining.add((s, m, "F"))
            remaining.add((s, m, "B"))

    done = set()
    result = []
    while remaining:
        cycle = [None] * stages
        for s in range(stages):
            candidates = []
            for op in remaining:
                if op[0] != s:
                    continue
                if all(d in done for d in deps(op)):
                    candidates.append(op)
            if candidates:
                candidates.sort(key=lambda x: (0 if x[2] == "B" else 1, x[1]))
                cycle[s] = candidates[0]
        if all(x is None for x in cycle):
            raise RuntimeError("oracle found a deadlock")
        for op in cycle:
            if op is not None:
                done.add(op)
                remaining.remove(op)
        result.append(cycle)
    return result


def grade(sol, fx) -> dict:
    cases = [
        (2, 2),
        (3, 3),
        (4, 2),
        (3, 5),
    ]
    ok = 1.0
    for stages, microbatches in cases:
        try:
            got = sol.pipeline_schedule(stages, microbatches)
            got = [[tuple(x) if x is not None else None for x in row] for row in got]
        except Exception:
            ok = 0.0
            break
        if got != _oracle(stages, microbatches):
            ok = 0.0
            break
    return {"exact_match": ok}
