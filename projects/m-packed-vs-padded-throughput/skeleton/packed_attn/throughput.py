def padded_flops(
    batch_seqlens: list[int],
    num_heads: int,
    head_dim: int,
    max_seqlen: int | None = None,
) -> int:
    raise NotImplementedError


def packed_flops(
    batch_seqlens: list[int], num_heads: int, head_dim: int
) -> int:
    raise NotImplementedError


def flops_ratio(
    batch_seqlens: list[int],
    num_heads: int,
    head_dim: int,
    max_seqlen: int | None = None,
) -> float:
    raise NotImplementedError


def memory_bytes(
    batch_seqlens: list[int],
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
    max_seqlen: int | None = None,
) -> dict[str, int]:
    raise NotImplementedError
