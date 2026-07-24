def _oracle_mro_names(cls):
    return [c.__name__ for c in cls.__mro__]


def grade(sol, fx) -> dict:
    class A:
        pass

    class B(A):
        pass

    class C(A):
        pass

    class D(B, C):
        pass

    class Left:
        pass

    class Right:
        pass

    class Mix(Left, Right):
        pass

    class Root:
        pass

    class L1(Root):
        pass

    class L2(Root):
        pass

    class Mid(L1, L2):
        pass

    class Diamond(Mid, L2):
        pass

    cases = [D, Mix, Diamond]

    ok = 1.0
    for cls in cases:
        try:
            got = sol.diamond_merge(cls)
        except Exception:
            ok = 0.0
            break
        if list(got) != _oracle_mro_names(cls):
            ok = 0.0
            break
    return {"exact_match": ok}
