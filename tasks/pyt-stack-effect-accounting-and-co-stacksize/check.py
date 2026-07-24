import dis


def _oracle(source):
    code = compile(source, "<oracle>", "exec")
    depth = 0
    max_depth = 0
    net = 0
    for ins in dis.get_instructions(code):
        effect = dis.stack_effect(ins.opcode, ins.arg)
        net += effect
        depth += effect
        if depth > max_depth:
            max_depth = depth
    return (int(net), int(max_depth))


def grade(sol, fx) -> dict:
    cases = [
        "x = 1 + 2",
        "a = [1, 2, 3]\nb = a[0]",
        "total = 0\nfor x in range(5):\n    total += x",
        "def f(x):\n    return x * 2\n",
        "d = {i: i*i for i in range(4)}",
    ]
    ok = 1.0
    for source in cases:
        try:
            got = sol.stack_account(source)
            got = (int(got[0]), int(got[1]))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(source):
            ok = 0.0
            break
    return {"exact_match": ok}
