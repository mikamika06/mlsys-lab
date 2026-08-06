def pipeline_schedule(stages, microbatches):
    def deps(op):
        s, m, phase = op
        if phase == "F":
            return [] if s == 0 else [(s - 1, m, "F")]
        return [(s, m, "F")] + ([] if s == stages - 1 else [(s + 1, m, "B")])

    remaining = set()
    for s in range(stages):
        for m in range(microbatches):
            for p in ("F", "B"):
                remaining.add((s, m, p))

    done = set()
    schedule = []

    while remaining:
        cycle = [None] * stages
        for s in range(stages):
            available = []
            for op in remaining:
                if op[0] == s:
                    d_list = deps(op)
                    all_done = True
                    for dep in d_list:
                        if dep not in done:
                            all_done = False
                            break
                    if all_done:
                        available.append(op)
            if available:
                n = len(available)
                for i in range(n):
                    for j in range(0, n - i - 1):
                        op1 = available[j]
                        op2 = available[j + 1]
                        k1 = 0 if op1[2] == "B" else 1
                        k2 = 0 if op2[2] == "B" else 1
                        if (k1, op1[1]) > (k2, op2[1]):
                            available[j], available[j + 1] = available[j + 1], available[j]
                cycle[s] = available[0]
        for op in cycle:
            if op is not None:
                remaining.remove(op)
                done.add(op)
        schedule.append(cycle)

    return schedule
