import numpy as np


def mxfp4_full_block_quantize(W: np.ndarray) -> dict:
    """MXFP4: W is (B, 32). Per block, compute shared power-of-two scale
    2**e with e = max(0, ceil(log2(max|w|/6))), snap w/scale onto the
    E2M1 grid {0,0.5,1,1.5,2,3,4,6} (signed), dequantize by * scale.
    Return {"scale": (B,), "codes": (B,32), "dequant": (B,32)}."""
    raise NotImplementedError('your code here')
