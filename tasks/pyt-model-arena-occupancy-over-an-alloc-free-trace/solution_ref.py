def arena_occupancy(trace, pools_per_arena, blocks_per_pool):
    arenas = []
    owners = {}
    result = []

    for op, ident in trace:
        if op == "alloc":
            located = False
            for ai, arena in enumerate(arenas):
                for pi, used in enumerate(arena):
                    if used < blocks_per_pool:
                        arena[pi] += 1
                        owners[ident] = (ai, pi)
                        located = True
                        break
                if located:
                    break

            if not located:
                for ai, arena in enumerate(arenas):
                    if len(arena) < pools_per_arena:
                        arena.append(1)
                        owners[ident] = (ai, len(arena) - 1)
                        located = True
                        break

            if not located:
                arenas.append([1])
                owners[ident] = (len(arenas) - 1, 0)

        elif op == "free":
            ai, pi = owners.pop(ident)
            arenas[ai][pi] -= 1

        result.append(len(arenas))

    return result
