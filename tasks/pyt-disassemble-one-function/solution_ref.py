import dis


def disassemble_one_function(fn):
    return [instruction.opname for instruction in dis.get_instructions(fn)]
