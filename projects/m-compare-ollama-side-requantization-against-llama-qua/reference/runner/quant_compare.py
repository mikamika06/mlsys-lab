import numpy as np


def llama_quantize_mock(tensor_data, quant_type="Q4_0"):
    """Simulate llama-quantize block-wise quantization."""
    arr = np.array(tensor_data, dtype=np.float32)
    block_size = 32
    padded_len = int(np.ceil(arr.size / block_size)) * block_size
    padded = np.pad(arr.ravel(), (0, padded_len - arr.size), mode="constant")
    blocks = padded.reshape(-1, block_size)
    dequantized = np.zeros_like(blocks)

    if quant_type == "Q4_0":
        max_vals = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales = np.where(max_vals == 0, 1.0, max_vals / 7.0)
        q = np.clip(np.round(blocks / scales), -8, 7)
        dequantized = q * scales
    elif quant_type == "Q8_0":
        max_vals = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales = np.where(max_vals == 0, 1.0, max_vals / 127.0)
        q = np.clip(np.round(blocks / scales), -128, 127)
        dequantized = q * scales
    else:
        raise ValueError(f"Unsupported quant_type: {quant_type}")

    return dequantized.ravel()[: arr.size].reshape(arr.shape)


def ollama_requantize_mock(tensor_data, quant_type="Q4_0"):
    """Simulate Ollama-side requantization."""
    arr = np.array(tensor_data, dtype=np.float32)
    block_size = 32
    padded_len = int(np.ceil(arr.size / block_size)) * block_size
    padded = np.pad(arr.ravel(), (0, padded_len - arr.size), mode="constant")
    blocks = padded.reshape(-1, block_size)
    dequantized = np.zeros_like(blocks)

    if quant_type == "Q4_0":
        max_vals = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales = np.where(max_vals == 0, 1.0, max_vals / 7.5)
        q = np.clip(np.floor(blocks / scales + 0.5), -8, 7)
        dequantized = q * scales
    elif quant_type == "Q8_0":
        max_vals = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales = np.where(max_vals == 0, 1.0, max_vals / 127.0)
        q = np.clip(np.floor(blocks / scales + 0.5), -128, 127)
        dequantized = q * scales
    else:
        raise ValueError(f"Unsupported quant_type: {quant_type}")

    return dequantized.ravel()[: arr.size].reshape(arr.shape)


def compare_quantization_drift(tensor_data, quant_type="Q4_0"):
    """Compare quantization MSE and max absolute diff between schemes."""
    arr = np.array(tensor_data, dtype=np.float32)
    q_llama = llama_quantize_mock(arr, quant_type=quant_type)
    q_ollama = ollama_requantize_mock(arr, quant_type=quant_type)

    mse_llama = float(np.mean((arr - q_llama) ** 2))
    mse_ollama = float(np.mean((arr - q_ollama) ** 2))
    max_diff = float(np.max(np.abs(q_llama - q_ollama)))
    cosine_sim = float(
        np.dot(q_llama.ravel(), q_ollama.ravel())
        / (np.linalg.norm(q_llama) * np.linalg.norm(q_ollama) + 1e-12)
    )

    return {
        "mse_llama": mse_llama,
        "mse_ollama": mse_ollama,
        "max_diff": max_diff,
        "cosine_sim": cosine_sim,
    }
