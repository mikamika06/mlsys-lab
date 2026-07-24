from mlsys.sim import cache as cachesim


def _simulate_branch_and_trace(values, threshold):
    state = 1
    mispredicts = 0
    addrs = []

    for i, x in enumerate(values):
        taken = x > threshold
        prediction = state >= 2
        if prediction != taken:
            mispredicts += 1

        if taken:
            state = min(3, state + 1)
        else:
            state = max(0, state - 1)

        addrs.append(0x1000 + i * 8)

    return mispredicts, addrs


def _reference(arr_sorted, arr_unsorted, threshold):
    sorted_count, sorted_trace = _simulate_branch_and_trace(arr_sorted, threshold)
    unsorted_count, unsorted_trace = _simulate_branch_and_trace(arr_unsorted, threshold)

    sorted_cache = cachesim.simulate(
        sorted_trace,
        line_bytes=16,
        sets=4,
        ways=2,
    )
    unsorted_cache = cachesim.simulate(
        unsorted_trace,
        line_bytes=16,
        sets=4,
        ways=2,
    )

    if sorted_cache["misses"] < 0 or unsorted_cache["misses"] < 0:
        raise RuntimeError("invalid cache simulation result")

    return sorted_count, unsorted_count


def grade(sol, fx) -> dict:
    cases = [
        ([1, 2, 3, 8, 9, 10], [8, 1, 9, 2, 10, 3], 5),
        ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], 2),
        ([10, 11, 12, 13], [10, 13, 11, 12], 9),
        ([-3, -1, 0, 4, 8], [8, 0, -3, 4, -1], 1),
    ]

    ok = 1.0
    for arr_sorted, arr_unsorted, threshold in cases:
        expected = _reference(
            list(arr_sorted),
            list(arr_unsorted),
            threshold,
        )
        try:
            got = tuple(
                sol.branch_mispredict_counts(
                    list(arr_sorted),
                    list(arr_unsorted),
                    threshold,
                )
            )
        except Exception:
            ok = 0.0
            break

        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
