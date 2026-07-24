import dis


def _oracle(code):
    depth = 0
    out = []
    for instr in dis.get_instructions(code):
        depth += dis.stack_effect(instr.opcode, instr.arg)
        out.append(depth)
    return out


def grade(sol, fx) -> dict:
    def f1(x):
        return x + 1

    def f2(a, b):
        c = a * b
        return c - 3

    def f3(xs):
        return [x * 2 for x in xs]

    cases = [
        f1.__code__,
        f2.__code__,
        f3.__code__,
        (lambda: {"a": 1, "b": 2}).__code__,
    ]

    ok = 1.0
    for code in cases:
        try:
            got = sol.stack_depth_timeline(code)
        except Exception:
            ok = 0.0
            break
        if list(got) != _oracle(code):
            ok = 0.0
            break
    return {"exact_match": ok}
