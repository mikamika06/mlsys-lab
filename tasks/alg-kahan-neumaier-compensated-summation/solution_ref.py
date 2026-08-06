import struct


def compensated_sum(arr: list[float]) -> float:
    def f32(v: float) -> float:
        return struct.unpack('f', struct.pack('f', v))[0]

    s = 0.0
    c = 0.0
    for x in arr:
        t = f32(s + x)
        abs_s = s if s >= 0 else -s
        abs_x = x if x >= 0 else -x
        if abs_s >= abs_x:
            c = f32(c + f32(f32(s - t) + x))
        else:
            c = f32(c + f32(f32(x - t) + s))
        s = t
    return f32(s + c)
