CONFIGS = [
    (8, 4),
    (16, 4),
    (64, 8),
    (128, 16),
    (1024, 64)
]


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


def workload_imbalance(assignment: list[list[int]]) -> dict[str, float]:
    workloads = [sum(i + 1 for i in dev_blocks) for dev_blocks in assignment]
    mean_w = sum(workloads) / len(workloads)
    max_w = max(workloads)
    return {
        "mean": float(mean_w),
        "max": float(max_w),
        "rel_err": float((max_w - mean_w) / mean_w) if mean_w > 0 else 0.0
    }
