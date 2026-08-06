import numpy as np


def fp32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    x_f32 = np.asarray(x, dtype=np.float32)
    u = x_f32.view(np.uint32)
    nan_mask = np.isnan(x_f32)

    lsb = (u >> 16) & 1
    bias = np.uint32(0x7FFF) + lsb
    rounded = ((u.astype(np.uint64) + bias.astype(np.uint64)) >> 16).astype(np.uint16)

    if np.any(nan_mask):
        nan_bits = ((u[nan_mask] >> 16) | np.uint32(0x0040)).astype(np.uint16)
        rounded[nan_mask] = nan_bits

    return rounded


def bf16_bits_to_fp32(bits: np.ndarray) -> np.ndarray:
    bits_u16 = np.asarray(bits, dtype=np.uint16)
    u32 = bits_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def round_fp32_to_bf16(x: np.ndarray) -> np.ndarray:
    bits = fp32_to_bf16_bits(x)
    return bf16_bits_to_fp32(bits)
