import sys


def _oracle(steps):
    class Obj:
        pass

    obj = Obj()
    aliases = []
    container = []
    returned = None

    def consume(value):
        return None

    timeline = []
    for step in steps:
        if step == "assign":
            aliases.append(obj)
        elif step == "alias":
            aliases.append(obj)
        elif step == "container-insert":
            container.append(obj)
        elif step == "function-arg":
            consume(obj)
        elif step == "return":
            returned = obj
        else:
            raise ValueError(step)
        timeline.append(sys.getrefcount(obj) - 1)
    return timeline


def grade(sol, fx) -> dict:
    cases = [
        [
            "assign",
            "alias",
            "container-insert",
            "function-arg",
            "return",
        ],
        [
            "function-arg",
            "assign",
            "function-arg",
            "container-insert",
        ],
        [
            "alias",
            "alias",
            "return",
            "function-arg",
        ],
        [
            "container-insert",
            "assign",
            "return",
            "alias",
            "function-arg",
        ],
    ]

    ok = 1.0
    for steps in cases:
        try:
            got = sol.refcount_timeline(list(steps))
            expected = _oracle(list(steps))
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
