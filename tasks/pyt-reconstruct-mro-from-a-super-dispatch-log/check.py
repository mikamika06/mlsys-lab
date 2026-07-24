def _make_log(cls):
    return [f"super_dispatch:{c.__name__}" for c in cls.__mro__ if c is not object]


def _oracle(cls):
    return tuple(c.__name__ for c in cls.__mro__)


def grade(sol, fx) -> dict:
    class A:
        pass

    class B(A):
        pass

    class C(A):
        pass

    class D(B, C):
        pass

    class X:
        pass

    class Y:
        pass

    class Z(X, Y):
        pass

    class M(Z, D):
        pass

    class Single:
        pass

    cases = [D, Z, M, Single]

    ok = 1.0
    for cls in cases:
        try:
            got = tuple(sol.reconstruct_mro(_make_log(cls)))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(cls):
            ok = 0.0
            break

    return {"exact_match": ok}
