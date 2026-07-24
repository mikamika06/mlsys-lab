def _oracle(buffers, world_size):
    chunk_size = len(buffers[0]) // world_size
    result = []
    for owner in range(world_size):
        chunk = [0] * chunk_size
        start = owner * chunk_size
        end = start + chunk_size
        for buf in buffers:
            part = buf[start:end]
            for i, value in enumerate(part):
                chunk[i] += value
        result.append(chunk)
    return result


def grade(sol, fx) -> dict:
    cases = [
        (
            [
                [1, 2, 10, 20],
                [3, 4, 30, 40],
            ],
            2,
        ),
        (
            [
                [5, 6, 7, 8, 9, 10],
                [1, 2, 3, 4, 5, 6],
                [10, 20, 30, 40, 50, 60],
            ],
            3,
        ),
        (
            [
                [4, -1, 8, 2],
                [7, 3, 0, 5],
            ],
            2,
        ),
    ]

    ok = 1.0
    for buffers, world_size in cases:
        expected = _oracle(buffers, world_size)
        try:
            got = sol.reduce_scatter_chunks(
                [list(row) for row in buffers],
                world_size,
            )
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
