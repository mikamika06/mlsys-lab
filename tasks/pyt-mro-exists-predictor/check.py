def _oracle(graph):
    made = {}

    def make_class(index, stack):
        if index in made:
            return made[index]
        if index in stack:
            raise TypeError("cyclic inheritance")
        stack = set(stack)
        stack.add(index)
        bases = tuple(make_class(base, stack) for base in graph[index])
        made[index] = type("Node%d" % index, bases, {})
        return made[index]

    try:
        type("Target", tuple(make_class(base, set()) for base in graph[0]), {})
        return True
    except TypeError:
        return False


def grade(sol, fx) -> dict:
    cases = [
        [[1, 2], [], []],
        [[1, 2], [3], [3], []],
        [[1, 2], [3, 4], [4, 3], [], []],
        [[1, 2, 3], [], [], []],
        [[1, 2], [3], [2], []],
        [[1, 2, 3], [3, 2], [], []],
        [[1, 2], [3, 4], [5], [5], [5], []],
        [[1, 2], [3], [4], [4], []],
        [[1, 2, 3, 4], [], [], [], []],
        [[1, 2], [3, 4], [4, 5], [5], [6], [], []],
        [[1, 2], [3], [4], [5], [5], [], []],
        [[1, 2], [2, 3], [3], []],
    ]

    ok = 1.0
    for graph in cases:
        try:
            got = bool(sol.mro_exists([list(x) for x in graph]))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(graph):
            ok = 0.0
            break
    return {"exact_match": ok}
