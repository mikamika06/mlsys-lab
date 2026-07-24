def _oracle(memory_budget, n_ctx, block_size, request_lengths):
    slot_used = 0
    slot_capacity = 0
    for _ in request_lengths:
        slot_used += n_ctx
        if slot_used <= memory_budget:
            slot_capacity += 1
        else:
            break

    paged_used = 0
    paged_capacity = 0
    for length in request_lengths:
        blocks = (length + block_size - 1) // block_size
        paged_used += blocks * block_size
        if paged_used <= memory_budget:
            paged_capacity += 1
        else:
            break

    return slot_capacity, paged_capacity


def grade(sol, fx) -> dict:
    cases = [
        (100, 30, 8, [7, 12, 20, 4]),
        (256, 64, 16, [1, 1, 1, 33, 64, 2]),
        (512, 128, 32, [100, 20, 31, 32, 33, 64]),
        (75, 20, 7, [6, 14, 15, 1, 50]),
        (1024, 256, 64, list(range(1, 40))),
    ]

    ok = 1.0
    for memory_budget, n_ctx, block_size, request_lengths in cases:
        expected = _oracle(memory_budget, n_ctx, block_size, request_lengths)
        try:
            got = sol.admission_capacity(
                memory_budget,
                n_ctx,
                block_size,
                list(request_lengths),
            )
            got = tuple(got)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
