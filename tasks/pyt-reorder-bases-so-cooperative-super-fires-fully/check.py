from itertools import permutations


def _oracle(bases):
    def evaluate(order):
        try:
            class Combined(*order):
                def run(self):
                    return super().run()
            return Combined().run()
        except Exception:
            return None

    for order in permutations(bases):
        result = evaluate(order)
        if result is not None:
            tags = [cls._tag for cls in order]
            if result[:len(tags)] == tags and "Root" in result:
                return tuple(order), result
    raise RuntimeError("no valid order")


def grade(sol, fx) -> dict:
    cases = []

    class Root:
        def run(self):
            return ["Root"]

    Root._tag = "Root"

    class A(Root):
        def run(self):
            return ["A"] + super().run()

    A._tag = "A"

    class B(A):
        def run(self):
            return ["B"] + super().run()

    B._tag = "B"

    class X(Root):
        def run(self):
            return ["X"] + super().run()

    X._tag = "X"

    class Y(X):
        def run(self):
            return ["Y"] + super().run()

    Y._tag = "Y"

    cases.append([A, B])
    cases.append([X, Y])
    cases.append([A, B, Y])

    ok = 1.0
    for bases in cases:
        try:
            got_order = tuple(sol.reorder_bases(list(bases)))
            class Combined(*got_order):
                def run(self):
                    return super().run()
            got_result = Combined().run()
            ref_order, ref_result = _oracle(bases)
            if got_order != ref_order or got_result != ref_result:
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
