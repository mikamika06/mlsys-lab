def _oracle(n_threads, n_increments):
    from functools import lru_cache

    @lru_cache(None)
    def solve(counter, threads):
        threads = tuple(threads)
        if all(t[0] == n_increments for t in threads):
            return frozenset([counter])

        outcomes = set()
        for idx, state in enumerate(threads):
            done, phase, reg = state
            if done == n_increments:
                continue

            if phase == 0:
                new_state = (done, 1, counter)
                nxt = list(threads)
                nxt[idx] = new_state
                outcomes.update(solve(counter, tuple(nxt)))
            elif phase == 1:
                new_state = (done, 2, reg + 1)
                nxt = list(threads)
                nxt[idx] = new_state
                outcomes.update(solve(counter, tuple(nxt)))
            else:
                new_state = (done + 1, 0, 0)
                nxt = list(threads)
                nxt[idx] = new_state
                outcomes.update(solve(reg, tuple(nxt)))
        return frozenset(outcomes)

    initial = tuple((0, 0, 0) for _ in range(n_threads))
    return set(solve(0, initial))


def grade(sol, fx) -> dict:
    cases = [
        (2, 1),
        (2, 2),
        (3, 1),
        (3, 2),
    ]

    ok = 1.0
    for n_threads, n_increments in cases:
        try:
            got = sol.race_outcomes(n_threads, n_increments)
            got = set(got)
        except Exception:
            ok = 0.0
            break

        ref = _oracle(n_threads, n_increments)
        if got != ref:
            ok = 0.0
            break

    return {"exact_match": ok}
