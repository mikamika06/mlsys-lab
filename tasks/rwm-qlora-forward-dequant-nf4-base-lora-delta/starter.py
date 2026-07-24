import numpy as np

NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=np.float64)


def qlora_forward(
    x: np.ndarray,
    nf4_codes: np.ndarray,
    absmax: np.ndarray,
    blocksize: int,
    A: np.ndarray,
    B: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    QLoRA forward pass: dequantize the NF4 base weight (blockwise absmax),
    add the scaled LoRA delta B@A, then apply the resulting linear layer.
    See task.md for the exact three-step formula.
    """
    raise NotImplementedError('your code here')
