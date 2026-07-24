import sys


def _oracle_ratio(n):
    values = [x for x in range(n)]
    generator = (x for x in range(n))
    return sys.getsizeof(values) / sys.getsizeof(generator)


def grade(sol, fx) -> dict:
    cases = [0, 1, 10, 100, 1000, 10000]
    scores = []
    for n in cases:
        try:
            got = float(sol.footprint_ratio(n))
            ref = _oracle_ratio(n)
        except Exception:
            return {"size_ratio": 0.0}
        if got <= 0 or ref <= 0:
            scores.append(0.0)
        else:
            scores.append(min(got / ref, ref / got))
    return {"size_ratio": sum(scores) / len(scores)}
