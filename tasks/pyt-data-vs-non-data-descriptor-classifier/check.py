def _oracle(classes):
    result = []
    for cls in classes:
        result.append(
            hasattr(cls, "__set__") or hasattr(cls, "__delete__")
        )
    return result


def grade(sol, fx) -> dict:
    class GetOnly:
        def __get__(self, obj, owner):
            return 1

    class SetOnly:
        def __set__(self, obj, value):
            pass

    class DeleteOnly:
        def __delete__(self, obj):
            pass

    class GetSet:
        def __get__(self, obj, owner):
            return 1
        def __set__(self, obj, value):
            pass

    class GetDelete:
        def __get__(self, obj, owner):
            return 1
        def __delete__(self, obj):
            pass

    class GetSetDelete:
        def __get__(self, obj, owner):
            return 1
        def __set__(self, obj, value):
            pass
        def __delete__(self, obj):
            pass

    class Empty:
        pass

    class GetAndOther:
        def __get__(self, obj, owner):
            return 1
        def other(self):
            pass

    class SetAndOther:
        def __set__(self, obj, value):
            pass
        def other(self):
            pass

    class DeleteAndOther:
        def __delete__(self, obj):
            pass
        def other(self):
            pass

    class FakeSetName:
        __set_name__ = lambda self, owner, name: None

    class ClassOnly:
        @classmethod
        def __get__(cls, obj, owner):
            return 1

    class StaticGet:
        __get__ = staticmethod(lambda obj, owner: 1)

    class PropertyLike:
        def __get__(self, obj, owner):
            return 1
        def __set_name__(self, owner, name):
            pass

    class BothSpecial:
        def __get__(self, obj, owner):
            return 1
        def __delete__(self, obj):
            pass
        def __set_name__(self, owner, name):
            pass

    cases = [
        GetOnly,
        SetOnly,
        DeleteOnly,
        GetSet,
        GetDelete,
        GetSetDelete,
        Empty,
        GetAndOther,
        SetAndOther,
        DeleteAndOther,
        FakeSetName,
        ClassOnly,
        StaticGet,
        PropertyLike,
        BothSpecial,
    ]

    try:
        got = list(sol.classify_descriptors(cases))
    except Exception:
        return {"exact_match": 0.0}

    ref = _oracle(cases)
    return {"exact_match": float(got == ref)}
