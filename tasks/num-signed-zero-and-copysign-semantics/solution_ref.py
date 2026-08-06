import math


def signed_zero_profile():
    values = [
        math.copysign(float("inf"), -1.0),
        -0.0 + 0.0,
        0.0 + -0.0,
        math.copysign(0.0, -1.0),
        math.copysign(-0.0, 1.0),
        math.copysign(5.0, -0.0),
    ]
    result = []
    for x in values:
        if math.copysign(1.0, x) < 0:
            result.append(1)
        else:
            result.append(0)
    return result
