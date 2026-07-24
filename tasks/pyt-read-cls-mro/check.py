def _oracle(cls):
    return tuple(c.__name__ for c in cls.__mro__)


def grade(sol, fx) -> dict:
    class Root:
        pass

    class A(Root):
        pass

    class B(Root):
        pass

    class C(A, B):
        pass

    class D(C):
        pass

    class Left:
        pass

    class Right:
        pass

    class Diamond(Left, Right):
        pass

    cases = [Root, A, C, D, Diamond]

    ok = 1.0
    for cls in cases:
        try:
            got = tuple(sol.mro_names(cls))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(cls):
            ok = 0.0
            break

    return {"exact_match": ok}
