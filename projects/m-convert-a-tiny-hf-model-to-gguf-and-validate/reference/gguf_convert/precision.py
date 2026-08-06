import numpy as np


def cast_tensor_to_outtype(array: np.ndarray, outtype: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if outtype == "f32":
        return arr.copy()
    elif outtype == "f16":
        return arr.astype(np.float16).astype(np.float32)
    elif outtype == "bf16":
        raw32 = arr.view(np.uint32)
        bf16_bits = (raw32 >> 16) & 0xFFFF
        shifted = bf16_bits.astype(np.uint32) << 16
        return shifted.view(np.float32)
    else:
        raise ValueError(f"Unsupported outtype: {outtype}")


def compute_representation_error(state_dict: dict, outtype: str) -> dict:
    max_err = 0.0
    total_sq_err = 0.0
    total_count = 0

    for _, tensor in state_dict.items():
        arr = np.asarray(tensor, dtype=np.float32)
        quantized = cast_tensor_to_outtype(arr, outtype)
        diff = np.abs(arr - quantized)

        curr_max = float(np.max(diff)) if diff.size > 0 else 0.0
        if curr_max > max_err:
            max_err = curr_max

        total_sq_err += float(np.sum((arr - quantized) ** 2))
        total_count += arr.size

    mse = total_sq_err / total_count if total_count > 0 else 0.0
    return {
        "max_abs_error": float(max_err),
        "mse": float(mse)
    }
