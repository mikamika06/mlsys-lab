import math

def rms_norm(x: list[list[float]], eps: float=1e-05) -> list[list[float]]:
    """
    Incorrect implementation that subtracts the mean before normalizing.
    This mimics a LayerNorm style centering and is not what RMSNorm should do.
    """
    raise NotImplementedError('your code here')
