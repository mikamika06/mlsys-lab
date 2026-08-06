def ulysses_comm_volume_per_layer(
    seq_len: int, hidden_dim: int, world_size: int, dtype_bytes: int = 2
) -> int:
    """Calculate total Ulysses communication volume per layer in bytes."""
    if world_size <= 1:
        return 0
    per_rank_bytes = (seq_len // world_size) * hidden_dim * dtype_bytes
    alltoall_1_vol = 2 * (world_size - 1) * per_rank_bytes
    alltoall_2_vol = 2 * (world_size - 1) * per_rank_bytes
    return 2 * (alltoall_1_vol + alltoall_2_vol)


def ring_comm_volume_per_layer(
    seq_len: int, hidden_dim: int, world_size: int, dtype_bytes: int = 2
) -> int:
    """Calculate total Ring Attention communication volume per layer in bytes."""
    if world_size <= 1:
        return 0
    k_v_bytes = 2 * (seq_len // world_size) * hidden_dim * dtype_bytes
    p2p_comm = (world_size - 1) * k_v_bytes
    fwd_comm = 2 * p2p_comm
    bwd_comm = 2 * p2p_comm
    return fwd_comm + bwd_comm


def usp_comm_volume_per_layer(
    seq_len: int,
    hidden_dim: int,
    world_size: int,
    ulysses_degree: int,
    ring_degree: int,
    dtype_bytes: int = 2,
) -> int:
    """Calculate total USP hybrid communication volume per layer in bytes."""
    if ulysses_degree * ring_degree != world_size:
        raise ValueError("ulysses_degree * ring_degree must equal world_size")
    u_vol = ulysses_comm_volume_per_layer(
        seq_len, hidden_dim, ulysses_degree, dtype_bytes
    )
    r_vol = ring_comm_volume_per_layer(
        seq_len // ulysses_degree, hidden_dim, ring_degree, dtype_bytes
    )
    return u_vol + r_vol
