def _oracle_chain(x):
    chain = []
    current = type(x)
    while True:
        name = current.__name__
        chain.append(name)
        if name == "type":
            break
        current = type(current)
    return chain


def grade(sol, fx) -> dict:
    class UserClass:
        pass

    class Meta(type):
        pass

    class CustomClass(metaclass=Meta):
        pass

    cases = [
        42,
        "arena",
        [],
        {},
        UserClass(),
        UserClass,
        CustomClass(),
        CustomClass,
        type,
    ]

    total = len(cases)
    matched = 0.0
    for obj in cases:
        try:
            got = sol.metatype_chain(obj)
        except Exception:
            continue
        if list(got) == _oracle_chain(obj):
            matched += 1.0

    return {"exact_match": matched / total}
