import struct


def tree_sum(values: list[float]) -> float:
    current = [struct.unpack('f', struct.pack('f', v))[0] for v in values]

    while len(current) > 1:
        if len(current) % 2 == 1:
            padded = [0.0] * (len(current) + 1)
            for i in range(len(current)):
                padded[i] = current[i]
            padded[len(current)] = struct.unpack('f', struct.pack('f', 0.0))[0]
            current = padded

        new_size = len(current) // 2
        next_current = [0.0] * new_size
        for i in range(new_size):
            val = current[2 * i] + current[2 * i + 1]
            next_current[i] = struct.unpack('f', struct.pack('f', val))[0]
        current = next_current

    if len(current) == 0:
        return 0.0
    return struct.unpack('f', struct.pack('f', current[0]))[0]
