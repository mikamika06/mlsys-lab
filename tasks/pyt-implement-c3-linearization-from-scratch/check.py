import numpy as np


def _graph_for(cls):
    classes = list(cls.__mro__)
    index = {c: i for i, c in enumerate(classes)}
    graph = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for i, c in enumerate(classes):
        for base in c.__bases__:
            if base in index:
                graph[i, index[base]] = 1
    names = [c.__name__ for c in classes]
    return graph, names, 0


def _build_cases():
    class A:
        pass

    class B(A):
        pass

    class C(A):
        pass

    class D(B, C):
        pass

    class E(A):
        pass

    class F(E, A):
        pass

    class G(D, E):
        pass

    class H(B):
        pass

    class I(C, H):
        pass

    return [D, G, I]


def grade(sol, fx) -> dict:
    ok = 1.0
    for cls in _build_cases():
        try:
            graph, names, idx = _graph_for(cls)
            got = sol.c3(graph, idx, names)
        except Exception:
            ok = 0.0
            break
        ref = [c.__name__ for c in cls.__mro__]
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
