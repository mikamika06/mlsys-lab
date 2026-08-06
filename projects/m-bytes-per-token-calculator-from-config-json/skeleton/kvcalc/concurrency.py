def max_concurrency(vram_budget_bytes: int, bytes_per_tok: int, max_seq_len: int, overhead_bytes: int = 0) -> int:
    """Calculate maximum safe request concurrency under VRAM budget."""
    raise NotImplementedError
