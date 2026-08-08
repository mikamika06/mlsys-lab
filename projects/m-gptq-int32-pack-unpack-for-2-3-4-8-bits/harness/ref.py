import numpy as np

def get_test_matrices():
    np.random.seed(42)
    matrices = []
    for bits in [2, 3, 4, 8]:
        shape = (16, 16)
        max_val = (1 << bits) - 1
        mat = np.random.randint(0, max_val + 1, size=shape, dtype=np.int32)
        matrices.append((mat, bits))
    return matrices

def pack_weights(matrix, bits):
    flat = matrix.flatten().astype(np.int32)
    mask = (1 << bits) - 1
    flat = flat & mask
    values_per_int = 32 // bits
    if bits == 3:
        values_per_int = 10
        padded_len = ((len(flat) + values_per_int - 1) // values_per_int) * values_per_int
        padded = np.zeros(padded_len, dtype=np.int32)
        padded[:len(flat)] = flat
        res = []
        for i in range(0, len(padded), values_per_int):
            chunk = padded[i:i+values_per_int]
            val = 0
            for shift_idx, v in enumerate(chunk):
                val |= int(v) << (shift_idx * 3)
            res.append(val)
        return np.array(res, dtype=np.int32)
    else:
        padded_len = ((len(flat) + values_per_int - 1) // values_per_int) * values_per_int
        padded = np.zeros(padded_len, dtype=np.int32)
        padded[:len(flat)] = flat
        reshaped = padded.reshape(-1, values_per_int)
        packed = np.zeros(reshaped.shape[0], dtype=np.int32)
        for i in range(values_per_int):
            packed |= reshaped[:, i] << (i * bits)
        return packed

def unpack_weights(packed, bits, original_shape):
    total_elements = np.prod(original_shape)
    if bits == 3:
        values_per_int = 10
        unpacked = []
        for val in packed:
            for shift_idx in range(values_per_int):
                v = (val >> (shift_idx * 3)) & 7
                unpacked.append(v)
        flat = np.array(unpacked, dtype=np.int32)[:total_elements]
        return flat.reshape(original_shape)
    else:
        values_per_int = 32 // bits
        mask = (1 << bits) - 1
        unpacked = []
        for val in packed:
            for i in range(values_per_int):
                v = (val >> (i * bits)) & mask
                unpacked.append(v)
        flat = np.array(unpacked, dtype=np.int32)[:total_elements]
        return flat.reshape(original_shape)

def convert_awq_to_gptq(awq_packed, bits, original_shape):
    unpacked = unpack_weights(awq_packed, bits, original_shape)
    transposed = unpacked.T
    return pack_weights(transposed, bits)

def packed_shape_and_stride(shape, bits):
    rows, cols = shape
    values_per_int = 32 // bits
    if bits == 3:
        packed_rows = (rows * cols + 9) // 10
        return (packed_rows, 1), (1, 1)
    else:
        packed_rows = (rows * cols + values_per_int - 1) // values_per_int
        return (packed_rows, 1), (1, 1)
