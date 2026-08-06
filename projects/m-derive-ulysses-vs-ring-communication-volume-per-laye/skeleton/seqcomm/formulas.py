def ulysses_comm_volume_per_layer(
    seq_len: int, hidden_dim: int, world_size: int, dtype_bytes: int = 2
) -> int:
    """Calculate total Ulysses communication volume per layer in bytes."""
    raise NotImplementedError


def ring_comm_volume_per_layer(
    seq_len: int, hidden_dim: int, world_size: int, dtype_bytes: int = 2
) -> int:
    """Calculate total Ring Attention communication volume per layer in bytes."""
    raise NotImplementedError


def usp_comm_volume_per_layer(
    seq_len: int,
    hidden_dim: int,
    world_size: int,
    ulysses_degree: int,
    ring_degree: int,
    dtype_bytes: int = 2,
) -> int:
    """Calculate total USP hybrid communication volume per layer in bytes."""
    raise NotImplementedError
