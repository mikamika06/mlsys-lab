import numpy as np

TEST_WEIGHTS_SYM = [
    np.array([-5.0, -2.0, 1.0, 4.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    np.array([-128.0, 64.0, 32.0, -64.0], dtype=np.float32)
]

TEST_WEIGHTS_AFF = [
    np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    np.array([-10.0, 0.0, 10.0, 20.0], dtype=np.float32)
]

SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (4096, 4096)
]


def derive_symmetric(weights, bits=4):
    qmax = (1 << (bits - 1)) - 1
    max_val = np.max(np.abs(weights))
    scale = max_val / qmax if max_val != 0 else 1.0
    zero_point = 0
    return float(scale), int(zero_point)


def derive_affine(weights, bits=4):
    qmin = 0
    qmax = (1 << bits) - 1
    min_val = float(np.min(weights))
    max_val = float(np.max(weights))
    if min_val == max_val:
        scale = 1.0
        zero_point = qmin
    else:
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = int(np.round(qmin - min_val / scale))
        zero_point = int(np.clip(zero_point, qmin, qmax))
    return float(scale), int(zero_point)


def blockwise_size_ratio(shape, block_size, bits=4):
    total_elements = int(np.prod(shape))
    num_blocks = (total_elements + block_size - 1) // block_size
    weight_bits = total_elements * bits
    overhead_bits = num_blocks * 32
    compressed_bits = weight_bits + overhead_bits
    uncompressed_bits = total_elements * 16
    return float(compressed_bits / uncompressed_bits)


def block_size_sweep(weights, block_sizes, bits=4):
    flat = weights.flatten()
    n = len(flat)
    errors = []
    for bs in block_sizes:
        total_err = 0.0
        count = 0
        for i in range(0, n, bs):
            block = flat[i:i + bs]
            scale, zp = derive_symmetric(block, bits=bits)
            qmin = -(1 << (bits - 1))
            qmax = (1 << (bits - 1)) - 1
            quantized = np.clip(np.round(block / scale) + zp, qmin, qmax)
            dequantized = (quantized - zp) * scale
            total_err += float(np.sum((block - dequantized) ** 2))
            count += len(block)
        mse = total_err / count if count > 0 else 0.0
        errors.append((bs, mse))
    return errors
