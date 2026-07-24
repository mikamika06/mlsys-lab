def _oracle(values, index, new_value):
    buf = bytearray(values)
    first = memoryview(buf)
    second = memoryview(buf)

    before = second[index]
    first[index] = new_value

    return before, first[index], second[index]


def grade(sol, fx) -> dict:
    cases = [
        ([10, 20, 30], 1, 99),
        ([0, 1, 255, 4], 2, 17),
        ([7, 8, 9, 10, 11], 4, 200),
        ([42], 0, 1),
    ]

    ok = 1.0
    for values, index, new_value in cases:
        try:
            got = tuple(sol.prove_memory_sharing(list(values), index, new_value))
        except Exception:
            ok = 0.0
            break

        if got != _oracle(list(values), index, new_value):
            ok = 0.0
            break

    return {"exact_match": ok}
