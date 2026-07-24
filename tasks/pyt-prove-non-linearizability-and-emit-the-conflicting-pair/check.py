import re


def _oracle(spec):
    classes = {}
    for name, bases in spec:
        if not bases:
            classes[name] = type(name, (), {})
        else:
            classes[name] = type(name, tuple(classes[b] for b in bases), {})
    raise AssertionError("expected a non-linearizable hierarchy")


def _oracle_pair(spec):
    classes = {}
    try:
        for name, bases in spec:
            if not bases:
                classes[name] = type(name, (), {})
            else:
                classes[name] = type(name, tuple(classes[b] for b in bases), {})
    except TypeError as exc:
        match = re.search(r"bases ([A-Za-z_]\w*), ([A-Za-z_]\w*)", str(exc))
        if match:
            return (match.group(1), match.group(2))
    return None


def grade(sol, fx) -> dict:
    cases = [
        [
            ("A", ()),
            ("B", ()),
            ("X", ("A", "B")),
            ("Y", ("B", "A")),
            ("Z", ("X", "Y")),
        ],
        [
            ("O", ()),
            ("P", ()),
            ("Q", ("O", "P")),
            ("R", ("P", "O")),
            ("S", ("Q", "R")),
        ],
        [
            ("Left", ()),
            ("Right", ()),
            ("L1", ("Left", "Right")),
            ("R1", ("Right", "Left")),
            ("Top", ("L1", "R1")),
        ],
    ]
    ok = 1.0
    for spec in cases:
        expected = _oracle_pair(spec)
        try:
            got = sol.conflicting_pair(spec)
        except Exception:
            ok = 0.0
            break
        if tuple(got) != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
