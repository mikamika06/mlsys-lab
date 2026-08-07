import numpy as np


def generate_test_tensor():
    rng = np.random.default_rng(1337)
    tensor = rng.standard_normal((64, 64)).astype(np.float32)
    tensor[10, 5] = 50.0
    tensor[20, 12] = -45.0
    return tensor


def reference_quantize(tensor, block_size):
    flat = tensor.astype(np.float32).flatten()
    padded_len = int(np.ceil(len(flat) / block_size) * block_size)
    padded = np.zeros(padded_len, dtype=np.float32)
    padded[:len(flat)] = flat
    blocks = padded.reshape(-1, block_size)
    absmax = np.max(np.abs(blocks), axis=1, keepdims=True)
    absmax[absmax == 0.0] = 1.0
    scaled = np.round(blocks / absmax * 127.0)
    quantized = np.clip(scaled, -128, 127).astype(np.int8)
    return {"quantized": quantized, "absmax": absmax.flatten(), "original_shape": tensor.shape}


def reference_outliers(tensor, threshold=5.0):
    mean = np.mean(tensor, axis=0)
    std = np.std(tensor, axis=0)
    std[std == 0.0] = 1.0
    z_scores = np.abs((tensor - mean) / std)
    column_max_z = np.max(z_scores, axis=0)
    return column_max_z > threshold


def reference_dequantize(quant_dict, block_size):
    quantized = quant_dict["quantized"].astype(np.float32)
    absmax = quant_dict["absmax"][:, np.newaxis]
    dequant_blocks = (quantized / 127.0) * absmax
    flattened = dequant_blocks.flatten()
    orig_shape = quant_dict["original_shape"]
    total_elements = int(np.prod(orig_shape))
    return flattened[:total_elements].reshape(orig_shape)
