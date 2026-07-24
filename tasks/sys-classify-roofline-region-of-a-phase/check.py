import numpy as np


def _roofline_oracle(configs):
    labels = []
    for item in configs:
        values = np.asarray(item, dtype=np.float64)
        batch = values[0]
        seq = values[1]
        dim = values[2]
        ridge = values[3]

        flops = np.float64(2.0) * batch * seq * seq * dim
        flops += np.float64(2.0) * batch * seq * dim * dim

        bytes_moved = np.float64(4.0) * (
            batch * seq * dim + batch * dim * dim
        )

        intensity = flops / bytes_moved

        if intensity < ridge:
            labels.append("bandwidth-bound")
        else:
            labels.append("compute-bound")

    return labels


def grade(sol, fx) -> dict:
    cases = [
        [
            (1, 2048, 4096, 80.0),
            (1, 32, 4096, 80.0),
        ],
        [
            (8, 512, 1024, 50.0),
            (4, 128, 2048, 120.0),
            (2, 64, 4096, 200.0),
        ],
        [
            (1, 1, 4096, 100.0),
            (16, 1024, 8192, 60.0),
            (4, 256, 1024, 64.0),
        ],
        [
            (2, 256, 512, 20.0),
            (2, 256, 512, 200.0),
            (32, 16, 2048, 10.0),
        ],
    ]

    expected = _roofline_oracle(cases[0] + cases[1] + cases[2] + cases[3])

    try:
        got = sol.classify_roofline_region(cases[0] + cases[1] + cases[2] + cases[3])
    except Exception:
        return {"exact_match": 0.0}

    return {"exact_match": float(got == expected)}
