def pipeline_schedule(stages, microbatches):
    # TODO: broken ordering. It always chooses forward operations first,
    # which can delay ready backward operations and differs from the required
    # 1F1B priority policy.
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
    out = []

    while remaining:
        cycle = [None] * stages
        for s in range(stages):
            available = [
                op for op in remaining
                if op[0] == s and all(dep in done for dep in deps(op))
            ]
            if available:
                available.sort(key=lambda op: (0 if op[2] == "F" else 1, op[1]))
                cycle[s] = available[0]
        if all(x is None for x in cycle):
            break
        for op in cycle:
            if op is not None:
                remaining.remove(op)
                done.add(op)
        out.append(cycle)

    return out
