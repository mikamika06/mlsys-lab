import math


def pick_collective(message_sizes, processes, alpha, beta):
    result = []
    for m in message_sizes:
        tree = (
            alpha * math.ceil(math.log2(processes))
            + beta * m * (processes - 1) / processes
        )
        ring = (
            alpha * (processes - 1)
            + beta * m * (2 * (processes - 1)) / processes
        )
        result.append("tree" if tree <= ring else "ring")
    return result
