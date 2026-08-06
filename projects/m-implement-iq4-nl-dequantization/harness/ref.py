import numpy as np

IQ4_NL_CODEBOOK = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32) / 127.0

def ref_dequantize_iq4_nl(data: bytes, scales: np.ndarray) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    low = arr & 0x0F
    high = (arr >> 4) & 0x0F
    unpacked = np.empty(arr.size * 2, dtype=np.float32)
    unpacked[0::2] = IQ4_NL_CODEBOOK[low]
    unpacked[1::2] = IQ4_NL_CODEBOOK[high]
    return unpacked * scales

def ref_decode_iq4_xs(superblock: bytes) -> np.ndarray:
    scale = float(superblock[0]) * 0.01
    payload = np.frombuffer(superblock[1:], dtype=np.uint8).astype(np.float32)
    return scale * (payload - 128.0)

def ref_compute_bpw(quant_type: str) -> float:
    table = {
        "IQ1_S": 1.3125,
        "IQ2_XXS": 2.25,
        "IQ4_XS": 4.25,
        "TQ1_0": 1.0,
        "TQ2_0": 2.0
    }
    return table[quant_type]
