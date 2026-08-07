import math


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
    mode_clean = mode.lower()
    if mode_clean not in ("dense", "ulysses", "ring"):
        raise ValueError(f"Unknown mode: {mode}")

    l = float(num_layers)
    h = float(hidden_dim)
    nh = float(num_heads)
    p = float(world_size)
    e = float(bytes_per_elem)
    s = float(seq_len)

    if mode_clean == "dense":
        c1 = l * c_lin * h * e
        c2 = l * nh * e
    elif mode_clean == "ulysses":
        c1 = (l * c_lin * h * e) / p
        c2 = (l * nh * e) / p
    else:
        c1 = (l * c_lin * h * e) / p
        c2 = (l * nh * e) / (p * p)

    return float(c1 * s + c2 * (s**2))


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
    avail = float(memory_budget_bytes) - float(model_bytes_per_rank)
    if avail <= 0:
        return 0

    mode_clean = mode.lower()
    l = float(num_layers)
    h = float(hidden_dim)
    nh = float(num_heads)
    p = float(world_size)
    e = float(bytes_per_elem)

    if mode_clean == "dense":
        c1 = l * c_lin * h * e
        c2 = l * nh * e
    elif mode_clean == "ulysses":
        c1 = (l * c_lin * h * e) / p
        c2 = (l * nh * e) / p
    elif mode_clean == "ring":
        c1 = (l * c_lin * h * e) / p
        c2 = (l * nh * e) / (p * p)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if c2 == 0.0:
        s = avail / c1
    else:
        disc = c1 * c1 + 4.0 * c2 * avail
        s = (-c1 + math.sqrt(disc)) / (2.0 * c2)

    return int(math.floor(s))


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
    return {
        "dense": max_sequence_length(
            memory_budget_bytes, model_bytes_per_rank, num_layers,
            hidden_dim, num_heads, world_size, "dense", c_lin, bytes_per_elem
        ),
        "ulysses": max_sequence_length(
            memory_budget_bytes, model_bytes_per_rank, num_layers,
            hidden_dim, num_heads, world_size, "ulysses", c_lin, bytes_per_elem
        ),
        "ring": max_sequence_length(
            memory_budget_bytes, model_bytes_per_rank, num_layers,
            hidden_dim, num_heads, world_size, "ring", c_lin, bytes_per_elem
        ),
    }
