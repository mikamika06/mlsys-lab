from mlsys.sim import cache as cachesim


def _ref(instructions):
    raw = 0
    war = 0
    waw = 0
    for i in range(len(instructions)):
        reads_i = set(instructions[i]["reads"])
        writes_i = set(instructions[i]["writes"])
        for j in range(i + 1, len(instructions)):
            reads_j = set(instructions[j]["reads"])
            writes_j = set(instructions[j]["writes"])
            if writes_i & reads_j:
                raw += 1
            if reads_i & writes_j:
                war += 1
            if writes_i & writes_j:
                waw += 1
    return raw, war, waw


def grade(sol, fx) -> dict:
    _ = cachesim.simulate(
        [0, 8, 16, 0, 24, 8],
        line_bytes=8,
        sets=2,
        ways=2,
    )

    cases = [
        [
            {"reads": [], "writes": [1]},
            {"reads": [1], "writes": [2]},
            {"reads": [2], "writes": [1]},
        ],
        [
            {"reads": [0], "writes": [3]},
            {"reads": [3], "writes": [3]},
            {"reads": [3], "writes": [4]},
            {"reads": [4], "writes": []},
        ],
        [
            {"reads": [1, 2], "writes": [5]},
            {"reads": [5], "writes": [2]},
            {"reads": [2], "writes": [5]},
            {"reads": [], "writes": [5]},
        ],
        [
            {"reads": [], "writes": []},
            {"reads": [7], "writes": []},
            {"reads": [], "writes": [7]},
        ],
    ]

    ok = 1.0
    for instructions in cases:
        try:
            got = sol.detect_hazards(instructions)
            got = tuple(got)
        except Exception:
            ok = 0.0
            break
        if got != _ref(instructions):
            ok = 0.0
            break
    return {"exact_match": ok}
