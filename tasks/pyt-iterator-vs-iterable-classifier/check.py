def _oracle(objects):
    result = []
    for obj in objects:
        try:
            result.append(iter(obj) is obj)
        except TypeError:
            result.append(False)
    return result


def grade(sol, fx) -> dict:
    def gen():
        yield 1

    cases = [
        [[], [1, 2], (1, 2), "abc"],
        [iter([1, 2]), iter((3, 4)), iter("xy")],
        [gen(), (x for x in range(2))],
        [42, None, object(), {"a": 1}, {1, 2}],
    ]

    expected = []
    got = []
    try:
        for case in cases:
            expected.extend(_oracle(case))
            got.extend(sol.classify_iterators(case))
    except Exception:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0 if got == expected else 0.0}
