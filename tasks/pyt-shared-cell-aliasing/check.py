def _oracle():
    def outer():
        value = 10

        def mutate():
            nonlocal value
            value += 3
            return value

        def observe():
            return value

        return mutate, observe

    mutate, observe = outer()
    values = []
    values.append(observe())
    values.append(mutate())
    values.append(observe())
    values.append(mutate())
    values.append(observe())
    return tuple(values)


def grade(sol, fx) -> dict:
    expected = _oracle()
    try:
        got = sol.shared_cell_trace()
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if got == expected else 0.0}
