import dis


def stack_account(source: str) -> tuple[int, int]:
    code = compile(source, "<stack>", "exec")
    depth = 0
    max_depth = 0
    net = 0

    for ins in dis.get_instructions(code):
        effect = dis.stack_effect(ins.opcode, ins.arg)
        net += effect
        depth += effect
        if depth > max_depth:
            max_depth = depth

    return int(net), int(max_depth)
