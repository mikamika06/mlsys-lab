def partition_flat_contiguous(
    tensor_sizes: list[int],
    world_size: int,
    alignment: int = 1,
) -> dict:
    """Partition flattened contiguous parameter buffer across DP ranks with alignment."""
    raise NotImplementedError


def partition_bin_packing(
    tensor_sizes: list[int],
    world_size: int,
) -> dict:
    """Bin-pack whole tensors into DP ranks using Longest Processing Time (LPT) heuristic."""
    raise NotImplementedError
