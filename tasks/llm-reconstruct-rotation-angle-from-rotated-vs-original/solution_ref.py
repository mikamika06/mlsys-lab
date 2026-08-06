import math

def recover_angles(orig: list[list[float]], rot: list[list[float]]) -> list[float]:
    """
    Compute the signed rotation angle (in radians) that maps each row of `orig`
    to the corresponding row of `rot` using explicit Python loops and math functions.
    Both inputs must be lists of length n containing lists of two floats. The output is a list
    of length n with floats.
    """
    angles = []
    for v, w in zip(orig, rot):
        vx, vy = v[0], v[1]
        wx, wy = w[0], w[1]
        cross = vx * wy - vy * wx
        dot = vx * wx + vy * wy
        angles.append(math.atan2(cross, dot))
    return angles
