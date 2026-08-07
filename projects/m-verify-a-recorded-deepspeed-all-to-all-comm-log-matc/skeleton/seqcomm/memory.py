def activation_memory_bytes(
    seq_len: int,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    world_size: int,
    mode: str,
    c_lin: float = 34.0,
    bytes_per_elem: int = 2,
) -> float:
    """Computes activation memory per rank in bytes for dense, ulysses, or ring mode."""
    raise NotImplementedError


def max_sequence_length(
    memory_budget_bytes: float,
    model_bytes_per_rank: float,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    world_size: int,
    mode: str,
    c_lin: float = 34.0,
    bytes_per_elem: int = 2,
) -> int:
    """Computes maximum sequence length achievable within memory budget."""
    raise NotImplementedError


def compare_sp_modes(
    memory_budget_bytes: float,
    model_bytes_per_rank: float,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    world_size: int,
    c_lin: float = 34.0,
    bytes_per_elem: int = 2,
) -> dict:
    """Compares max sequence length across dense, ulysses, and ring modes."""
    raise NotImplementedError
