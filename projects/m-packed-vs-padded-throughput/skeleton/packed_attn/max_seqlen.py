def tile_counts(
    batch_seqlens: list[int], tile_size: int, max_seqlen: int | None = None
) -> dict[str, int]:
    raise NotImplementedError


def throughput_ratio(
    batch_seqlens: list[int], tile_size: int, max_seqlen: int | None = None
) -> float:
    raise NotImplementedError


def mis_specification_penalty(
    batch_seqlens: list[int], tile_size: int, max_seqlen: int
) -> float:
    raise NotImplementedError
