import numpy as np


def _oracle(shards, shapes):
    gathered = np.concatenate([np.asarray(s).reshape(-1) for s in shards])
    total = sum(int(np.prod(shape)) for shape in shapes)
    flat = gathered[:total]
    result = []
    offset = 0
    for shape in shapes:
        count = int(np.prod(shape))
        result.append(flat[offset:offset + count].reshape(shape))
        offset += count
    return result


def grade(sol, fx) -> dict:
    cases = [
        (
            [
                np.array([1.5, -2.0, 3.0]),
                np.array([4.25, 5.5, 0.0]),
            ],
            [(2,), (3,)],
        ),
        (
            [
                np.arange(8, dtype=np.float64),
                np.array([8.0, 9.0, 10.0, -99.0]),
            ],
            [(2, 2), (3, 2)],
        ),
        (
            [
                np.array([0.25, 0.5, 0.75, 1.0]),
                np.array([-1.0, -2.0, -3.0, 123.0]),
                np.array([4.0, 5.0, 6.0, 7.0]),
            ],
            [(4,), (2, 3), (1, 2)],
        ),
    ]

    max_err = 0.0
    try:
        for shards, shapes in cases:
            expected = _oracle(shards, shapes)
            got = sol.unflatten_all_gathered(
                [np.array(s, copy=True) for s in shards],
                list(shapes),
            )
            if len(got) != len(expected):
                return {"max_abs_err": float("inf")}
            for a, b in zip(got, expected):
                err = float(np.max(np.abs(np.asarray(a) - b)))
                max_err = max(max_err, err)
    except Exception:
        max_err = float("inf")

    return {"max_abs_err": max_err}
