import numpy as np


def _oracle_triple(tag: str):
    if tag == "weight":
        return (-127, 127, True)  # qint8, symmetric, zero_point == 0
    if tag == "activation":
        return (0, 255, False)  # quint8, asymmetric, zero_point calibrated from data
    raise ValueError(f"unknown tensor tag: {tag!r}")


def _oracle(tags):
    return [_oracle_triple(t) for t in tags]


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    ok = 1.0

    for _ in range(6):
        n = int(rng.integers(3, 10))
        tags = [str(rng.choice(["weight", "activation"])) for _ in range(n)]
        exp = _oracle(tags)

        try:
            got = sol.ort_default_scheme(list(tags))
        except Exception:
            ok = 0.0
            continue

        try:
            got_norm = [(int(a), int(b), bool(c)) for (a, b, c) in got]
        except Exception:
            ok = 0.0
            continue

        if got_norm != exp:
            ok = 0.0

    return {"exact_match": ok}
