def _oracle(statement):
    import weakref

    freed = [False]
    refs = []

    class Temp:
        pass

    def on_free(_):
        freed[0] = True

    def make():
        obj = Temp()
        refs.append(weakref.ref(obj, on_free))
        return obj

    ns = {"make": make}
    exec(statement, ns, ns)
    return bool(freed[0])


def grade(sol, fx) -> dict:
    cases = [
        "make()",
        "x = make()",
        "items = [make()]",
        "pair = (make(),)",
        "d = {'value': make()}",
        "make(); y = 1",
        "a = make()\ndel a",
        "x = [make()]\nx.clear()",
        "x = make(); y = x",
        "x = make(); del x",
        "result = make()",
        "make().__class__",
        "obj = make()\nobj = None",
        "box = {'a': make()}\nbox = {}",
        "lst = []\nlst.append(make())",
        "t = make()\nkeep = t",
        "keep = make()\nkeep = keep",
        "make() or None",
        "x = (make() if True else None)",
    ]

    expected = []
    for stmt in cases:
        try:
            expected.append(_oracle(stmt))
        except Exception:
            return {"exact_match": 0.0}

    got = []
    try:
        for stmt in cases:
            got.append(bool(sol.classify_freed(stmt)))
    except Exception:
        return {"exact_match": 0.0}

    return {"exact_match": float(got == expected)}
