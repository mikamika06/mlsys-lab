import numpy as np

def convert_awq_to_gptq(packed_awq: np.ndarray, bits: int, rows: int, cols: int) -> np.ndarray:
    packed_awq = np.asarray(packed_awq, dtype=np.int32)
    unpacked = np.zeros(rows * cols, dtype=np.int32)
    vals = 32 // bits if bits != 3 else 10
    mask = (1 << bits) - 1 if bits != 3 else 7
    for i in range(vals):
        unpacked[i::vals] = (packed_awq >> (i * bits)) & mask
    reshaped = unpacked.reshape(rows, cols)
    transposed = reshaped.T
    flat = transposed.flatten()
    padded_len = (len(flat) + vals - 1) // vals * vals
    padded = np.zeros(padded_len, dtype=np.uint32)
    padded[:len(flat)] = flat
    reshaped_new = padded.reshape(-1, vals)
    packed = np.zeros(reshaped_new.shape[0], dtype=np.int32)
    for i in range(vals):
        packed |= (reshaped_new[:, i] & mask) << (i * bits)
    return packed
