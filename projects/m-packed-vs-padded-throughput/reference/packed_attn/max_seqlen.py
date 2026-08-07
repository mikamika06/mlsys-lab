import math


def tile_counts(
    batch_seqlens: list[int], tile_size: int, max_seqlen: int | None = None
) -> dict[str, int]:
    if not batch_seqlens:
        return {
            "ideal_packed_tiles": 0,
            "actual_packed_tiles": 0,
            "padded_tiles": 0,
        }
    actual_max = max(batch_seqlens)
    s_used = actual_max if max_seqlen is None else max_seqlen
    if s_used < actual_max:
        raise ValueError("max_seqlen cannot be less than max actual sequence length")

    ideal_packed = sum(
        math.ceil(length / tile_size) ** 2 for length in batch_seqlens
    )
    query_grid = math.ceil(s_used / tile_size)
    actual_packed = sum(
        query_grid * math.ceil(length / tile_size) for length in batch_seqlens
    )
    padded = len(batch_seqlens) * (query_grid**2)

    return {
        "ideal_packed_tiles": ideal_packed,
        "actual_packed_tiles": actual_packed,
        "padded_tiles": padded,
    }


def throughput_ratio(
    batch_seqlens: list[int], tile_size: int, max_seqlen: int | None = None
) -> float:
    tc = tile_counts(batch_seqlens, tile_size, max_seqlen)
    if tc["actual_packed_tiles"] == 0:
        return 0.0
    return float(tc["padded_tiles"]) / float(tc["actual_packed_tiles"])


def mis_specification_penalty(
    batch_seqlens: list[int], tile_size: int, max_seqlen: int
) -> float:
    tc = tile_counts(batch_seqlens, tile_size, max_seqlen)
    actual = tc["actual_packed_tiles"]
    if actual == 0:
        return 0.0
    ideal = tc["ideal_packed_tiles"]
    return 1.0 - (float(ideal) / float(actual))
