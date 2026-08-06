def validate_block_size(backend: dict, model: dict, block_size: int) -> dict:
    """Validate if a candidate block size satisfies backend and model alignment rules."""
    raise NotImplementedError


def filter_valid_block_sizes(backend: dict, model: dict, candidate_sizes: list) -> list:
    """Filter candidate block sizes to those passing all alignment checks."""
    raise NotImplementedError
