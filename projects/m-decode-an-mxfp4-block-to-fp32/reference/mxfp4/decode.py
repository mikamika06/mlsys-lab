import numpy as np

E2M1_TABLE = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=np.float32)


def decode_mxfp4_block(scale_e8m0: int, nibbles: np.ndarray) -> np.ndarray:
    """Decode a 32-element MXFP4 block using E2M1 FP4 elements and an E8M0 shared scale."""
    scale_e8m0 = int(scale_e8m0) & 0xFF
    if scale_e8m0 == 0:
        scale = 0.0
    else:
        scale = 2.0 ** (scale_e8m0 - 127)

    nibbles_arr = np.asarray(nibbles, dtype=np.uint8) & 0x0F
    fp4_vals = E2M1_TABLE[nibbles_arr]
    return (fp4_vals * scale).astype(np.float32)


def quantize_q4_0_block(values: np.ndarray) -> tuple[float, np.ndarray]:
    """Quantize 32 float32 values into Q4_0 format (fp16 scale + 32 4-bit signed ints)."""
    vals = np.asarray(values, dtype=np.float32)
    max_abs = np.max(np.abs(vals))
    if max_abs == 0.0:
        scale = 0.0
        q = np.full(vals.shape, 8, dtype=np.uint8)
        return scale, q

    scale = float(max_abs / 7.0)
    q_signed = np.clip(np.round(vals / scale), -8, 7)
    q_nibbles = (q_signed + 8).astype(np.uint8)
    return scale, q_nibbles


def decode_q4_0_block(scale: float, nibbles: np.ndarray) -> np.ndarray:
    """Decode 32 Q4_0 elements back to float32."""
    nibbles_arr = np.asarray(nibbles, dtype=np.uint8) & 0x0F
    q_signed = nibbles_arr.astype(np.float32) - 8.0
    return (q_signed * float(scale)).astype(np.float32)
