import numpy as np


def _oracle(grads, world_size):
    arr = np.asarray(grads, dtype=np.float64)
    reduced = np.sum(arr, axis=0)
    shard_size = reduced.shape[0] // world_size
    return [
        reduced[r * shard_size:(r + 1) * shard_size].tolist()
        for r in range(world_size)
    ]


def grade(sol, fx) -> dict:
    cases = [
        (
            [
                [1.0, 2.0, 3.0, 4.0],
                [10.0, 20.0, 30.0, 40.0],
            ],
            2,
        ),
        (
            [
                [1.0, 0.0, 2.0, 0.0, 3.0, 4.0],
                [5.0, 6.0, 0.0, 8.0, 9.0, 1.0],
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            ],
            3,
        ),
        (
            [
                [0.5, -1.5, 2.5, 4.5],
                [1.0, 2.0, -3.0, 8.0],
                [10.0, 20.0, 30.0, 40.0],
                [-2.0, 3.0, 1.0, 5.0],
            ],
            2,
        ),
    ]

    ok = 1.0
    for grads, world_size in cases:
        try:
            got = sol.reduce_scatter_owner(grads, world_size)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(grads, world_size):
            ok = 0.0
            break
    return {"exact_match": ok}
