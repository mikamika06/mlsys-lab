import numpy as np


NF4_LEVELS = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33795294165611267,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)


def _quantize_with_levels(tensor, levels, block_size=64):
    flat = tensor.astype(np.float32).flatten()
    numel = flat.size
    padded_size = ((numel + block_size - 1) // block_size) * block_size
    padded = np.zeros(padded_size, dtype=np.float32)
    padded[:numel] = flat
    blocks = padded.reshape(-1, block_size)

    absmaxes = np.max(np.abs(blocks), axis=1)
    absmaxes = np.where(absmaxes == 0, 1e-5, absmaxes)

    normalized = blocks / absmaxes[:, None]
    diffs = np.abs(normalized[..., None] - levels[None, None, :])
    codes = np.argmin(diffs, axis=2)
    reconstructed_norm = levels[codes]
    reconstructed_blocks = reconstructed_norm * absmaxes[:, None]
    reconstructed_flat = reconstructed_blocks.flatten()[:numel]

    mse = np.mean((flat - reconstructed_flat) ** 2)
    return {
        "codes": codes,
        "absmaxes": absmaxes,
        "reconstructed": reconstructed_flat.reshape(tensor.shape),
        "mse": float(mse),
    }


def quantize_nf4(tensor, block_size=64):
    return _quantize_with_levels(tensor, NF4_LEVELS, block_size)


def quantize_uniform_int4(tensor, block_size=64):
    uni_levels = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    return _quantize_with_levels(tensor, uni_levels, block_size)
