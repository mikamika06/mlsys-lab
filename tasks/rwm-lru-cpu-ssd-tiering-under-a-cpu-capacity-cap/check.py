from collections import OrderedDict


def _oracle(trace, sizes, savings, max_cpu_bytes):
    cpu = OrderedDict()
    used = 0

    for chunk in trace:
        if chunk in cpu:
            cpu.move_to_end(chunk)
        else:
            cpu[chunk] = None
            used += sizes[chunk]
            while used > max_cpu_bytes and cpu:
                old, _ = cpu.popitem(last=False)
                used -= sizes[old]

    resident = sorted(cpu.keys())
    total = sum(savings[x] for x in resident)
    return resident, total


def grade(sol, fx) -> dict:
    cases = [
        (
            [1, 2, 1, 3],
            {1: 4, 2: 6, 3: 5},
            {1: 10, 2: 7, 3: 8},
            10,
        ),
        (
            [4, 5, 6, 4, 5, 7],
            {4: 3, 5: 4, 6: 5, 7: 2},
            {4: 11, 5: 13, 6: 3, 7: 6},
            7,
        ),
        (
            [8, 9, 8, 10, 9, 8],
            {8: 5, 9: 5, 10: 5},
            {8: 20, 9: 15, 10: 12},
            10,
        ),
        (
            list(range(12)) + list(range(5, 12)),
            {i: (i % 4) + 1 for i in range(12)},
            {i: i * 3 + 1 for i in range(12)},
            9,
        ),
    ]

    score = 1.0
    for trace, sizes, savings, cap in cases:
        try:
            got = sol.tier_lru(list(trace), dict(sizes), dict(savings), cap)
            got = (list(got[0]), int(got[1]))
        except Exception:
            score = 0.0
            break
        if got != _oracle(trace, sizes, savings, cap):
            score = 0.0
            break

    return {"exact_match": score}
