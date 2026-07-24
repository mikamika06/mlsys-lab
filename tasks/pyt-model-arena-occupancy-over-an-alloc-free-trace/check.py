def _oracle(trace, pools_per_arena, blocks_per_pool):
    arenas = []
    owners = {}
    for op, ident in trace:
        if op == "alloc":
            placed = False
            for arena in arenas:
                for pool in arena:
                    if pool < blocks_per_pool:
                        arena[arena.index(pool)] += 1
                        owners[ident] = (arena, arena.index(pool))
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                for arena in arenas:
                    if len(arena) < pools_per_arena:
                        arena.append(1)
                        owners[ident] = (arena, len(arena) - 1)
                        placed = True
                        break
            if not placed:
                arenas.append([1])
                owners[ident] = (arenas[-1], 0)
        else:
            arena, index = owners.pop(ident)
            arena[index] -= 1
        # Resident arenas never disappear after creation.
    return_values = []
    return return_values


def _ref(trace, pools_per_arena, blocks_per_pool):
    arenas = []
    owners = {}
    out = []
    for op, ident in trace:
        if op == "alloc":
            found = False
            for a in range(len(arenas)):
                for p in range(len(arenas[a])):
                    if arenas[a][p] < blocks_per_pool:
                        arenas[a][p] += 1
                        owners[ident] = (a, p)
                        found = True
                        break
                if found:
                    break
            if not found:
                for a in range(len(arenas)):
                    if len(arenas[a]) < pools_per_arena:
                        arenas[a].append(1)
                        owners[ident] = (a, len(arenas[a]) - 1)
                        found = True
                        break
            if not found:
                arenas.append([1])
                owners[ident] = (len(arenas) - 1, 0)
        else:
            a, p = owners.pop(ident)
            arenas[a][p] -= 1
        out.append(len(arenas))
    return out


def grade(sol, fx) -> dict:
    cases = [
        ([("alloc", 1), ("alloc", 2), ("alloc", 3)], 2, 2),
        ([("alloc", 1), ("alloc", 2), ("alloc", 3), ("alloc", 4), ("alloc", 5)], 2, 2),
        ([("alloc", 1), ("alloc", 2), ("free", 1), ("alloc", 3), ("alloc", 4)], 3, 2),
        ([("alloc", 10), ("alloc", 11), ("free", 10), ("free", 11), ("alloc", 12)], 1, 2),
        ([("alloc", i) for i in range(12)], 3, 2),
    ]
    ok = 1.0
    for trace, p, b in cases:
        expected = _ref(trace, p, b)
        try:
            got = list(sol.arena_occupancy(trace, p, b))
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
