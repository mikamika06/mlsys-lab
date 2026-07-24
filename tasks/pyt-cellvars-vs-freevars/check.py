import types


def _oracle(source):
    code = compile(source, "<arena>", "exec")
    result = {}

    def walk(obj, prefix):
        if not isinstance(obj, types.CodeType):
            return
        name = prefix
        result[name] = {
            "co_cellvars": sorted(obj.co_cellvars),
            "co_freevars": sorted(obj.co_freevars),
        }
        for child in obj.co_consts:
            if isinstance(child, types.CodeType):
                child_name = child.co_name if name == "module" else name + "." + child.co_name
                walk(child, child_name)

    walk(code, "module")
    return result


def grade(sol, fx) -> dict:
    cases = [
        """
def outer():
    x = 1
    def inner():
        return x
    return inner
""",
        """
def factory(a):
    b = a + 1
    def first(c):
        def second():
            return b + c
        return second
    return first
""",
        """
def no_capture():
    value = 3
    def child():
        return 4
    return child
""",
        """
def mixed(flag):
    shared = 10
    def nested():
        if flag:
            return shared
        return flag
    return nested
""",
    ]

    ok = 1.0
    for source in cases:
        try:
            got = sol.analyze_closure(source)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(source):
            ok = 0.0
            break
    return {"exact_match": ok}
