import numpy as np

IQ4_NL_CODEBOOK = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32) / 127.0

def dequantize_iq4_nl(data: bytes, scales: np.ndarray) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    low = arr & 0x0F
    high = (arr >> 4) & 0x0F
    unpacked = np.empty(arr.size * 2, dtype=np.float32)
    unpacked[0::2] = IQ4_NL_CODEBOOK[low]
    unpacked[1::2] = IQ4_NL_CODEBOOK[high]
    return unpacked * scales
