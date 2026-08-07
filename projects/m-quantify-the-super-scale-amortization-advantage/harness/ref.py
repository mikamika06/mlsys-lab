import numpy as np

DATASETS = [
    np.random.RandomState(42).randn(1024).astype(np.float32),
    np.random.RandomState(123).randn(2048).astype(np.float32),
    np.sin(np.linspace(-10, 10, 4096)).astype(np.float32),
]

CONFIGS = [
    {"num_elements": 1024, "superblock_size": 256, "subblock_size": 32, "quant_bits": 4, "scale_bits": 6, "super_scale_bits": 16},
    {"num_elements": 2048, "superblock_size": 256, "subblock_size": 32, "quant_bits": 5, "scale_bits": 6, "super_scale_bits": 16},
    {"num_elements": 4096, "superblock_size": 128, "subblock_size": 16, "quant_bits": 6, "scale_bits": 8, "super_scale_bits": 16},
]


def compute_superblock_footprint(num_elements, superblock_size, subblock_size, quant_bits, scale_bits, super_scale_bits):
    num_superblocks = (num_elements + superblock_size - 1) // superblock_size
    num_subblocks_per_super = superblock_size // subblock_size

    quant_bits_per_super = superblock_size * quant_bits
    sub_scale_bits_per_super = num_subblocks_per_super * scale_bits
    super_scale_bits_per_super = 2 * super_scale_bits

    total_bits_per_super = quant_bits_per_super + sub_scale_bits_per_super + super_scale_bits_per_super
    total_bytes = num_superblocks * ((total_bits_per_super + 7) // 8)

    metadata_bits = num_superblocks * (sub_scale_bits_per_super + super_scale_bits_per_super)
    metadata_ratio = metadata_bits / (num_superblocks * total_bits_per_super)

    return {
        "num_superblocks": num_superblocks,
        "total_bytes": total_bytes,
        "metadata_ratio": metadata_ratio,
        "bits_per_weight": (total_bytes * 8) / num_elements
    }


def quantize_superblock(data, superblock_size, subblock_size, quant_bits):
    arr = np.asarray(data, dtype=np.float32)
    N = len(arr)
    num_superblocks = (N + superblock_size - 1) // superblock_size
    padded_len = num_superblocks * superblock_size
    if len(arr) < padded_len:
        arr = np.pad(arr, (0, padded_len - len(arr)), mode='constant', constant_values=0.0)

    num_subblocks_per_super = superblock_size // subblock_size
    reshaped = arr.reshape(num_superblocks, num_subblocks_per_super, subblock_size)

    d_super = np.max(np.abs(reshaped), axis=(1, 2), keepdims=True) / ((1 << quant_bits) - 1)
    d_super = np.maximum(d_super, 1e-10)

    sub_max = np.max(np.abs(reshaped), axis=2, keepdims=True)
    d_sub = np.clip(np.round(sub_max / d_super), 1, (1 << 6) - 1)

    effective_scale = d_super * d_sub
    max_quant = (1 << quant_bits) - 1
    quantized = np.clip(np.round((reshaped + np.abs(reshaped)) / (2.0 * effective_scale + 1e-12) * max_quant), 0, max_quant)
    dequantized = (quantized / max_quant * 2.0 - 1.0) * effective_scale

    reconstructed = dequantized.reshape(-1)[:N]
    mse = float(np.mean((arr[:N] - reconstructed) ** 2))
    return reconstructed, mse


def calculate_amortization_advantage(data, superblock_size, subblock_size, quant_bits):
    _, super_mse = quantize_superblock(data, superblock_size, subblock_size, quant_bits)
    _, uniform_mse = quantize_superblock(data, subblock_size, subblock_size, quant_bits)

    super_fp = compute_superblock_footprint(len(data), superblock_size, subblock_size, quant_bits, 6, 16)
    uniform_fp = compute_superblock_footprint(len(data), subblock_size, subblock_size, quant_bits, 0, 16)

    bytes_saved = uniform_fp["total_bytes"] - super_fp["total_bytes"]
    advantage_ratio = uniform_fp["metadata_ratio"] / max(super_fp["metadata_ratio"], 1e-10)

    return {
        "superblock_mse": super_mse,
        "uniform_mse": uniform_mse,
        "bytes_saved": bytes_saved,
        "advantage_ratio": advantage_ratio
    }
