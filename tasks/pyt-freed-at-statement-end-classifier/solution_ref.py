def classify_freed(statement):
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
