import math

def gathered_attention(k_phys: list[list[list[float]]], v_phys: list[list[list[float]]], block_table: list[int], seq_len: int, q: list[float]) -> list[float]:
    """Gather logical KV from a block table and compute single-query
    scaled dot-product attention."""
    raise NotImplementedError('your code here')
