import numpy as np

def pack_weights(weights: np.ndarray, bits: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.uint32)
    if bits == 3:
        vals = 10
        padded_len = (len(weights) + vals - 1) // vals * vals
        padded = np.zeros(padded_len, dtype=np.uint32)
        padded[:len(weights)] = weights
        reshaped = padded.reshape(-1, vals)
        packed = np.zeros(reshaped.shape[0], dtype=np.int32)
        for i in range(vals):
            packed |= (reshaped[:, i] & 7) << (i * 3)
        return packed
    else:
        vals = 32 // bits
        padded_len = (len(weights) + vals - 1) // vals * vals
        padded = np.zeros(padded_len, dtype=np.uint32)
        padded[:len(weights)] = weights
        reshaped = padded.reshape(-1, vals)
        packed = np.zeros(reshaped.shape[0], dtype=np.int32)
        mask = (1 << bits) - 1
        for i in range(vals):
            packed |= (reshaped[:, i] & mask) << (i * bits)
        return packed

def unpack_weights(packed: np.ndarray, bits: int, num_weights: int) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.int32)
    if bits == 3:
        vals = 10
        unpacked = np.zeros(len(packed) * vals, dtype=np.uint32)
        for i in range(vals):
            unpacked[i::vals] = (packed >> (i * 3)) & 7
        return unpacked[:num_weights].astype(np.int32)
    else:
        vals = 32 // bits
        unpacked = np.zeros(len(packed) * vals, dtype=np.uint32)
        mask = (1 << bits) - 1
        for i in range(vals):
            unpacked[i::vals] = (packed >> (i * bits)) & mask
        return unpacked[:num_weights].astype(np.int32)
