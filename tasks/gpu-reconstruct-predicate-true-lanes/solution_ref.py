def reconstruct_lanes(mask: int) -> list[int]:
    lanes = []
    for i in range(32):
        if mask & (1 << i):
            lanes.append(i)
    return lanes
