def _oracle(prefetch_depth):
    required_buffers = prefetch_depth + 1

    buffers = required_buffers
    ready = set()
    for step in range(prefetch_depth + 3):
        current = step % buffers
        if current not in ready and step >= prefetch_depth:
            return required_buffers, False

        ready.discard(current)

        for future in range(1, prefetch_depth + 1):
            ready.add((step + future) % buffers)

    return required_buffers, True


def grade(sol, fx) -> dict:
    cases = [0, 1, 2, 3, 5, 8, 13]
    for depth in cases:
        try:
            got = sol.kv_buffer_plan(depth)
        except Exception:
            return {"exact_match": 0.0}

        if tuple(got) != _oracle(depth):
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}
