import math


def streaming_log_softmax(x: list[float]) -> list[float]:
    m = -float("inf")
    for value in x:
        if value > m:
            m = value

    shifted = []
    for i in range(len(x)):
        shifted.append(x[i] - m)

    total = 0.0
    for value in shifted:
        total += math.exp(value)

    log_total = math.log(total)

    out = []
    for i in range(len(shifted)):
        out.append(shifted[i] - log_total)

    return out
