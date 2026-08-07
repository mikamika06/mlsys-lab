def allocate_pages(config, seq_len, block_size, page_align_bytes=64):
    """Compute per-layer allocated physical page bytes for a sequence length."""
    raise NotImplementedError
