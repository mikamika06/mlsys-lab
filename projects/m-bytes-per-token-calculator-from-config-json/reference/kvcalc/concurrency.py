def max_concurrency(vram_budget_bytes: int, bytes_per_tok: int, max_seq_len: int, overhead_bytes: int = 0) -> int:
    """Calculate maximum safe request concurrency under VRAM budget."""
    available = vram_budget_bytes - overhead_bytes
    if available <= 0 or bytes_per_tok <= 0 or max_seq_len <= 0:
        return 0
    per_req = bytes_per_tok * max_seq_len
    return int(available // per_req)
