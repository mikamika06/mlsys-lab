import dis


def stack_depth_timeline(code):
    depth = 0
    timeline = []
    for instr in dis.get_instructions(code):
        depth += dis.stack_effect(instr.opcode, instr.arg)
        timeline.append(depth)
    return timeline
