import numpy as np

def measure_quant_error(weights, format_name, block_size):
    flat = weights.flatten()
    n = len(flat)
    padded_len = ((n + block_size - 1) // block_size) * block_size
    padded = np.pad(flat, (0, padded_len - n), mode='constant')
    blocks = padded.reshape(-1, block_size)

    quantized_blocks = []
    for b in blocks:
        mx = np.max(np.abs(b))
        if mx == 0:
            scale = 1.0
        else:
            scale = mx / 7.0 if format_name in ["fp4", "int4"] else mx / 448.0

        if scale == 0:
            q = np.zeros_like(b)
        else:
            q = np.clip(np.round(b / scale), -7, 7) if format_name in ["fp4", "int4"] else np.clip(np.round(b / scale), -448, 448)
        dequant = q * scale
        quantized_blocks.append(dequant)

    reconstructed = np.concatenate(quantized_blocks)[:n]
    mse = np.mean((flat - reconstructed) ** 2)
    return float(mse)

def evaluate_end_to_end(weights, validation_data, format_name):
    mse = measure_quant_error(weights, format_name, block_size=32)
    score = float(np.mean(validation_data) * max(0.1, (1.0 - mse)))
    return score
