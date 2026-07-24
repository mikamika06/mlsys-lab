import dis


def opcode_sequence(func):
    return [ins.opname for ins in dis.get_instructions(func)]
