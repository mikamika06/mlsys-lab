import numpy as np

def dequantize_to_fp16(tensor_data, scale, zero_point):
    arr = np.array(tensor_data, dtype=np.float32)
    dequantized = (arr - zero_point) * scale
    return dequantized.astype(np.float16)
