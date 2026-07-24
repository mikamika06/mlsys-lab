import dis


def temporary_lifetime_step():
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
