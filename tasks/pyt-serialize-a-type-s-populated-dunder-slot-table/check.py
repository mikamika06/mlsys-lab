from mlsys import scorers


DUUNDERS = [
    "__new__", "__init__", "__repr__", "__str__", "__bytes__",
    "__format__", "__lt__", "__le__", "__eq__", "__ne__",
    "__gt__", "__ge__", "__hash__", "__bool__", "__len__",
    "__iter__", "__next__", "__getitem__", "__setitem__", "__delitem__",
    "__contains__", "__call__", "__enter__", "__exit__", "__await__",
    "__aiter__", "__anext__", "__add__", "__sub__", "__mul__",
    "__matmul__", "__truediv__", "__floordiv__", "__mod__",
    "__pow__", "__neg__", "__pos__", "__abs__", "__invert__",
    "__index__"
]


def _oracle(classes):
    out = bytearray()
    for cls in classes:
        mask = 0
        for i, name in enumerate(DUUNDERS):
            if getattr(cls, name, None) is not None:
                mask |= 1 << i
        out.extend(mask.to_bytes(8, "little", signed=False))
    return bytes(out)


def _make_cases():
    class A:
        def __len__(self):
            return 1

    class B(A):
        def __add__(self, other):
            return other

    class C:
        def __getitem__(self, x):
            return x

        def __iter__(self):
            return iter(())

    class D:
        def __call__(self):
            return 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class E:
        def __matmul__(self, other):
            return other

        def __index__(self):
            return 0

    class F:
        pass

    class G(int):
        def __repr__(self):
            return "g"

    class H(list):
        pass

    class I:
        def __eq__(self, other):
            return True

        def __hash__(self):
            return 1

    class J:
        def __await__(self):
            return iter(())

    return [A, B, C, D, E, F, G, H, I, J]


def grade(sol, fx) -> dict:
    try:
        cases = _make_cases()
        got = sol.serialize_dunder_slots(cases)
        ref = _oracle(cases)
        score = scorers.byte_exact_fraction(ref, got)
    except Exception:
        score = 0.0
    return {"byte_exact_fraction": score}
