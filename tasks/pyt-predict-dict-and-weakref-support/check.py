def _build_cases():
    cases = []

    class A:
        pass
    cases.append(A)

    class B:
        __slots__ = ()
    cases.append(B)

    class C:
        __slots__ = ("x",)
    cases.append(C)

    class D:
        __slots__ = ("__weakref__",)
    cases.append(D)

    class E:
        __slots__ = ("x", "__weakref__")
    cases.append(E)

    class F:
        __slots__ = ("__dict__",)
    cases.append(F)

    class G(A):
        __slots__ = ()
    cases.append(G)

    class H(B):
        pass
    cases.append(H)

    class I(B):
        __slots__ = ("__weakref__",)
    cases.append(I)

    class J(A):
        __slots__ = ("x",)
    cases.append(J)

    class K(C):
        __slots__ = ()
    cases.append(K)

    class L(C):
        __slots__ = ("__weakref__",)
    cases.append(L)

    class M(F):
        __slots__ = ()
    cases.append(M)

    class N(F):
        __slots__ = ("__weakref__",)
    cases.append(N)

    class O:
        __slots__ = ("a", "b", "__weakref__")
    cases.append(O)

    class P:
        __slots__ = ("__dict__", "__weakref__")
    cases.append(P)

    class Q(A):
        __slots__ = ()
    cases.append(Q)

    class R:
        __slots__ = ("a",)
    cases.append(R)

    class S(R):
        pass
    cases.append(S)

    class T(R):
        __slots__ = ("__dict__",)
    cases.append(T)

    return cases


def _oracle():
    import weakref

    out = []
    for cls in _build_cases():
        obj = cls()
        try:
            vars(obj)
            has_dict = True
        except TypeError:
            has_dict = False
        try:
            weakref.ref(obj)
            has_weakref = True
        except TypeError:
            has_weakref = False
        out.append([has_dict, has_weakref])
    return out


def grade(sol, fx) -> dict:
    try:
        got = sol.predict_layouts()
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if got == _oracle() else 0.0}
