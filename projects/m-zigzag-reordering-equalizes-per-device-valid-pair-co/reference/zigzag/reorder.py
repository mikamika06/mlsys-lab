def naive_assignment(num_blocks, num_devices):
    chunk = (num_blocks + num_devices - 1) // num_devices
    assignments = []
    for d in range(num_devices):
        start = d * chunk
        end = min(num_blocks, (d + 1) * chunk)
        assignments.append(list(range(start, end)))
    return assignments


def striped_assignment(num_blocks, num_devices):
    assignments = [[] for _ in range(num_devices)]
    for i in range(num_blocks):
        assignments[i % num_devices].append(i)
    return assignments


def zigzag_assignment(num_blocks, num_devices):
    assignments = [[] for _ in range(num_devices)]
    for i in range(num_blocks):
        round_idx = i // num_devices
        pos = i % num_devices
        if round_idx % 2 == 1:
            pos = num_devices - 1 - pos
        assignments[pos].append(i)
    return assignments
