def _oracle_trace(bases):
    name = "_ArenaC3Probe"
    probe = type(name, bases, {})
    return [cls.__name__ for cls in probe.mro()[1:]]


def grade(sol, fx) -> dict:
    cases = []

    class O:
        pass

    class A(O):
        pass

    class B(O):
        pass

    class C(A, B):
        pass

    class D(A):
        pass

    class E(B):
        pass

    class F(D, E):
        pass

    class X:
        pass

    class Y:
        pass

    class Z(X, Y):
        pass

    cases.extend([
        (A, B),
        (D, E),
        (F,),
        (Z,),
    ])

    ok = 1.0
    for bases in cases:
        try:
            got = sol.c3_merge_trace(bases)
        except Exception:
            ok = 0.0
            break
        if got != _oracle_trace(bases):
            ok = 0.0
            break
    return {"exact_match": ok}
