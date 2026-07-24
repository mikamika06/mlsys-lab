def race_outcomes(n_threads: int, n_increments: int) -> set[int]:
    from functools import lru_cache

    @lru_cache(None)
    def solve(counter, threads):
        if all(t[0] == n_increments for t in threads):
            return frozenset([counter])

        outcomes = set()
        for idx, (done, phase, reg) in enumerate(threads):
            if done == n_increments:
                continue

            nxt = list(threads)
            if phase == 0:
                nxt[idx] = (done, 1, counter)
                outcomes.update(solve(counter, tuple(nxt)))
            elif phase == 1:
                nxt[idx] = (done, 2, reg + 1)
                outcomes.update(solve(counter, tuple(nxt)))
            else:
                nxt[idx] = (done + 1, 0, 0)
                outcomes.update(solve(reg, tuple(nxt)))
        return frozenset(outcomes)

    initial = tuple((0, 0, 0) for _ in range(n_threads))
    return set(solve(0, initial))
