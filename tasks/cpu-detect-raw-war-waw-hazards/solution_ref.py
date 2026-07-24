def detect_hazards(instructions):
    raw = 0
    war = 0
    waw = 0

    for i in range(len(instructions)):
        reads_i = set(instructions[i]["reads"])
        writes_i = set(instructions[i]["writes"])
        for j in range(i + 1, len(instructions)):
            reads_j = set(instructions[j]["reads"])
            writes_j = set(instructions[j]["writes"])

            if writes_i & reads_j:
                raw += 1
            if reads_i & writes_j:
                war += 1
            if writes_i & writes_j:
                waw += 1

    return raw, war, waw
