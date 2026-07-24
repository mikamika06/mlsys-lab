def pipeline_schedule(stages, microbatches):
    def deps(op):
        s, m, phase = op
        if phase == "F":
            return [] if s == 0 else [(s - 1, m, "F")]
        return [(s, m, "F")] + ([] if s == stages - 1 else [(s + 1, m, "B")])

    remaining = {
        (s, m, p)
        for s in range(stages)
        for m in range(microbatches)
        for p in ("F", "B")
    }
    done = set()
    schedule = []

    while remaining:
        cycle = [None] * stages
        for s in range(stages):
            available = [
                op for op in remaining
                if op[0] == s and all(dep in done for dep in deps(op))
            ]
            if available:
                available.sort(key=lambda op: (0 if op[2] == "B" else 1, op[1]))
                cycle[s] = available[0]
        for op in cycle:
            if op is not None:
                remaining.remove(op)
                done.add(op)
        schedule.append(cycle)

    return schedule
