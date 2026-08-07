def expected_ulysses_comm(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    batch_size: int,
    world_size: int,
    bytes_per_elem: int = 2,
) -> dict:
    """Calculates expected communication stats for Ulysses sequence parallelism."""
    raise NotImplementedError


def verify_comm_log(records: list[dict], config: dict) -> dict:
    """Verifies recorded DeepSpeed all-to-all communication log entries."""
    raise NotImplementedError
