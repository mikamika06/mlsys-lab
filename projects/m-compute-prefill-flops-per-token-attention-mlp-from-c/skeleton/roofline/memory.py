"""Decode memory traffic computation."""


def compute_decode_bytes_per_step(config: dict, batch_size: int, context_len: int, dtype_bytes: int = 2) -> float:
    """Compute HBM memory bytes transferred per decode step."""
    raise NotImplementedError
