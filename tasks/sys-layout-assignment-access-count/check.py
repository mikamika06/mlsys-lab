def _ref(m, n, access_pattern, layout):
    """Compute the sequential-access count using the address formulas."""
    def addr(r, c):
        if layout == "row":
            return r * n + c
        else:
            return c * m + r
    if len(access_pattern) < 2:
        return 0
    addrs = [addr(r, c) for r, c in access_pattern]
    return sum(1 for i in range(len(addrs) - 1)
               if abs(addrs[i + 1] - addrs[i]) == 1)

def grade(sol, fx) -> dict:
    cases = [
        # (m, n, access_pattern, layout)
        (4, 4, [(0,0),(0,1),(0,2),(0,3),
                 (1,0),(1,1),(1,2),(1,3),
                 (2,0),(2,1),(2,2),(2,3),
                 (3,0),(3,1),(3,2),(3,3)], "row"),       # perfect row-major

        (4, 4, [(0,0),(0,1),(0,2),(0,3),
                 (1,0),(1,1),(1,2),(1,3),
                 (2,0),(2,1),(2,2),(2,3),
                 (3,0),(3,1),(3,2),(3,3)], "col"),        # stride-4 in col-major

        (4, 4, [(0,0),(1,0),(2,0),(3,0),
                 (0,1),(1,1),(2,1),(3,1),
                 (0,2),(1,2),(2,2),(3,2),
                 (0,3),(1,3),(2,3),(3,3)], "row"),        # stride-4 in row-major

        (4, 4, [(0,0),(1,0),(2,0),(3,0),
                 (0,1),(1,1),(2,1),(3,1),
                 (0,2),(1,2),(2,2),(3,2),
                 (0,3),(1,3),(2,3),(3,3)], "col"),        # perfect col-major

        (3, 5, [(0,0),(0,1),(0,2),(0,3),(0,4),
                 (1,0),(1,1),(1,2),(1,3),(1,4),
                 (2,0),(2,1),(2,2),(2,3),(2,4)], "row"),  # non-square row-major

        (3, 5, [(0,0),(0,1),(0,2),(0,3),(0,4),
                 (1,0),(1,1),(1,2),(1,3),(1,4),
                 (2,0),(2,1),(2,2),(2,3),(2,4)], "col"),  # non-square col-major

        (4, 4, [(0,0),(1,1),(2,2),(3,3)], "row"),         # diagonal, row-major

        (4, 4, [(0,0),(1,1),(2,2),(3,3)], "col"),         # diagonal, col-major

        (1, 1, [(0,0)], "row"),                           # single element

        (2, 3, [(0,0),(0,1)], "row"),                     # two-element pattern
    ]

    total = len(cases)
    correct = 0
    for m, n, pattern, layout in cases:
        try:
            got = sol.modeled_access_count(m, n, pattern, layout)
        except Exception:
            continue
        if got == _ref(m, n, pattern, layout):
            correct += 1

    return {"modeled_mem_access": correct / total if total else 0.0}
