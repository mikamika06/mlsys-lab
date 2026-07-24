def _ref(ops):
    names = [name for name, _ in ops]
    deps = {name: list(d) for name, d in ops}
    index = {name: i for i, name in enumerate(names)}
    remaining = set(names)
    out = []

    while remaining:
        ready = []
        for name in names:
            if name in remaining and all(dep in out for dep in deps[name]):
                ready.append(name)
        if not ready:
            return []
        chosen = ready[0]
        out.append(chosen)
        remaining.remove(chosen)

    return out


def _legal(schedule, ops):
    if len(schedule) != len(ops):
        return False
    if set(schedule) != {x[0] for x in ops}:
        return False
    pos = {name: i for i, name in enumerate(schedule)}
    for name, deps in ops:
        for dep in deps:
            if pos[dep] >= pos[name]:
                return False
    return True


def grade(sol, fx) -> dict:
    cases = [
        [
            ("comm_grad", ["compute_grad"]),
            ("compute_grad", []),
            ("update", ["comm_grad"]),
        ],
        [
            ("send", ["pack"]),
            ("pack", ["forward"]),
            ("forward", []),
            ("log", []),
        ],
        [
            ("reduce", ["allreduce_start"]),
            ("allreduce_start", ["kernel"]),
            ("kernel", []),
            ("copy", ["kernel"]),
        ],
        [
            ("b", []),
            ("a", []),
            ("c", ["a", "b"]),
            ("d", ["c"]),
        ],
    ]

    ok = 1.0
    for ops in cases:
        try:
            got = sol.fix_schedule(ops)
        except Exception:
            ok = 0.0
            break
        expected = _ref(ops)
        if got != expected or not _legal(got, ops):
            ok = 0.0
            break

    return {"exact_match": ok}
