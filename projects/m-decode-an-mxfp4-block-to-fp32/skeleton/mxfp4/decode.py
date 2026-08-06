import numpy as np


def decode_mxfp4_block(scale_e8m0: int, nibbles: np.ndarray) -> np.ndarray:
    """Decode a 32-element MXFP4 block using E2M1 FP4 elements and an E8M0 shared scale."""
    raise NotImplementedError


def quantize_q4_0_block(values: np.ndarray) -> tuple[float, np.ndarray]:
    """Quantize 32 float32 values into Q4_0 format (fp16 scale + 32 4-bit signed ints)."""
    raise NotImplementedError


def decode_q4_0_block(scale: float, nibbles: np.ndarray) -> np.ndarray:
    """Decode 32 Q4_0 elements back to float32."""
    raise NotImplementedError
