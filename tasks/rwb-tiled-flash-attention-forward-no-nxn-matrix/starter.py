import math

def tiled_flash_attention_forward(Q: list[list[float]], K: list[list[float]], V: list[list[float]], block_size: int=64) -> list[list[float]]:
    """Block-tiled flash-attention forward pass without materialising the full n x n score matrix."""
    raise NotImplementedError('your code here')
