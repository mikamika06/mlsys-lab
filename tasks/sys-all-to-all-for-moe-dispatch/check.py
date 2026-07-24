import numpy as np


def _oracle_all_to_all(send, world_size):
    received = []
    for dst in range(world_size):
        blocks = []
        for src in range(world_size):
            blocks.append(send[src, dst])
        received.append(np.concatenate(blocks, axis=0))
    return np.stack(received, axis=0)


def grade(sol, fx) -> dict:
    cases = [
        (2, 3, 2, 0.125),
        (3, 2, 4, 0.25),
        (4, 1, 3, 0.375),
        (3, 5, 2, 0.75),
    ]
    worst = 0.0
    for world_size, tokens, hidden, scale in cases:
        base = np.arange(
            world_size * world_size * tokens * hidden,
            dtype=np.float64,
        ).reshape(world_size, world_size, tokens, hidden)
        send = base * scale + 0.03125

        expected = _oracle_all_to_all(send, world_size)

        try:
            got = np.asarray(sol.moe_all_to_all(send, world_size))
            err = float(np.max(np.abs(got - expected)))
        except Exception:
            return {"max_abs_err": float("inf")}

        worst = max(worst, err)

    return {"max_abs_err": worst}
