def k_smallest_indices(arr: list[float], k: int) -> list[int]:
    # WRONG IMPLEMENTATION: Uses full sort!
    if k == 0:
        return []
    indexed = [(val, i) for i, val in enumerate(arr)]
    indexed.sort()
    return [i for val, i in indexed[:k]]
