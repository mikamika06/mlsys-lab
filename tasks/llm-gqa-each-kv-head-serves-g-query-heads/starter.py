def gqa_attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]], g: int) -> list[list[float]]:
    """Incorrect implementation that uses full attention over all KV heads.
This will fail the max_abs_err gate because it does not respect grouping."""
    raise NotImplementedError('your code here')
