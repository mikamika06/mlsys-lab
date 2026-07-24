import dis


def _oracle():
    def expr():
        class Temp:
            pass
        return (Temp(), 0)[1]

    instructions = list(dis.get_instructions(expr))
    saw_call = False
    for index, inst in enumerate(instructions):
        if inst.opname == "CALL":
            saw_call = True
        if saw_call and inst.opname == "BINARY_SUBSCR":
            return index
    raise RuntimeError("CPython bytecode pattern not found")


def grade(sol, fx) -> dict:
    expected = _oracle()
    try:
        got = sol.temporary_lifetime_step()
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if got == expected else 0.0}
