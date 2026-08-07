import numpy as np

FP4_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def quantize_nvfp4_block(
    tensor: np.ndarray, block_size: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    flat = tensor.astype(np.float64).flatten()
    n = flat.size
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len), mode="constant", constant_values=0.0)

    num_blocks = flat.size // block_size
    reshaped = flat.reshape(num_blocks, block_size)

    block_max = np.max(np.abs(reshaped), axis=1)
    scales = block_max / 6.0
    scales = np.where(scales == 0.0, 1.0, scales)

    scaled_reshaped = reshaped / scales[:, None]
    signs = np.sign(scaled_reshaped)
    signs = np.where(signs == 0, 1.0, signs)
    abs_scaled = np.abs(scaled_reshaped)

    diffs = np.abs(abs_scaled[:, :, None] - FP4_VALUES[None, None, :])
    mag_indices = np.argmin(diffs, axis=-1)

    codes = (np.where(signs < 0, 8, 0) | mag_indices).astype(np.uint8)
    return codes.reshape(-1)[:n], scales


def dequantize_nvfp4_block(
    codes: np.ndarray, scales: np.ndarray, block_size: int = 16
) -> np.ndarray:
    n = codes.size
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        codes_padded = np.pad(codes, (0, pad_len), mode="constant", constant_values=0)
    else:
        codes_padded = codes

    num_blocks = codes_padded.size // block_size
    reshaped_codes = codes_padded.reshape(num_blocks, block_size)

    sign_bits = (reshaped_codes >> 3) & 1
    mag_indices = reshaped_codes & 7

    mags = FP4_VALUES[mag_indices]
    signs = np.where(sign_bits == 1, -1.0, 1.0)
    dequant = signs * mags * scales[:, None]

    return dequant.reshape(-1)[:n]


def nvfp4_round_trip(tensor: np.ndarray, block_size: int = 16) -> np.ndarray:
    shape = tensor.shape
    codes, scales = quantize_nvfp4_block(tensor, block_size=block_size)
    dequant = dequantize_nvfp4_block(codes, scales, block_size=block_size)
    return dequant.reshape(shape)
