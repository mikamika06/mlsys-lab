def dict_probe_sequence(hash_value: int, size: int, steps: int) -> list[int]:
    mask = size - 1
    perturb = hash_value
    slot = hash_value & mask
    result = []

    for _ in range(steps):
        result.append(slot)
        slot = (5 * slot + 1 + perturb) & mask
        perturb >>= 5

    return result
