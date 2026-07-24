import sys


def _oracle(sizes):
    B = 32
    P = 64

    class_size = {}
    for n in sizes:
        if n not in class_size:
            class_size[n] = sys.getsizeof(bytearray(n))

    pools = {}
    free_pools = []
    arenas = []
    mmap_events = 0
    munmap_events = 0

    def create_arena():
        nonlocal mmap_events
        mmap_events += 1
        arena = {"active": 0}
        arenas.append(arena)
        free_pools.extend([arena] * P)

    def acquire_pool():
        if not free_pools:
            create_arena()
        arena = free_pools.pop()
        arena["active"] += 1
        return arena

    def release_pool(arena):
        nonlocal munmap_events
        arena["active"] -= 1
        free_pools.append(arena)
        if arena["active"] == 0:
            while arena in free_pools:
                free_pools.remove(arena)
            arenas.remove(arena)
            munmap_events += 1

    active = []
    for n in sizes:
        cls = class_size[n]
        if cls not in pools:
            pools[cls] = []
        if not pools[cls] or pools[cls][-1][1] == 0:
            pools[cls].append([acquire_pool(), B])
        pools[cls][-1][1] -= 1
        active.append((cls, pools[cls][-1]))

    for cls, pool in reversed(active):
        pool[1] += 1
        if pool[1] == B:
            pools[cls].pop()
            release_pool(pool[0])

    return mmap_events, munmap_events


def grade(sol, fx) -> dict:
    cases = [
        [1, 2, 3, 4, 5],
        [1] * 40,
        list(range(70)),
        [8] * 100 + [512] * 100 + [8] * 100,
        [0, 10, 20, 30, 1000, 2000, 1000, 30, 20, 10, 0],
    ]
    ok = 1.0
    for case in cases:
        try:
            got = tuple(sol.arena_mmap_events(list(case)))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(case):
            ok = 0.0
            break
    return {"exact_match": ok}
