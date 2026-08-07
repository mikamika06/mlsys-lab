def naive_assignment(num_blocks: int, num_devices: int) -> list[list[int]]:
    chunk = num_blocks // num_devices
    return [list(range(d * chunk, (d + 1) * chunk)) for d in range(num_devices)]


def striped_assignment(num_blocks: int, num_devices: int) -> list[list[int]]:
    out = [[] for _ in range(num_devices)]
    for i in range(num_blocks):
        out[i % num_devices].append(i)
    return out


def zigzag_assignment(num_blocks: int, num_devices: int) -> list[list[int]]:
    out = [[] for _ in range(num_devices)]
    for i in range(num_blocks):
        rem = i % (2 * num_devices)
        if rem < num_devices:
            dev = rem
        else:
            dev = (2 * num_devices - 1) - rem
        out[dev].append(i)
    return out
