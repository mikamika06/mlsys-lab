import numpy as np


def pack_tq1_0(values):
    vals = np.asarray(values, dtype=np.int8)
    n = len(vals)
    packed_bytes = bytearray()
    i = 0
    while i < n:
        chunk = vals[i:i+5]
        val = 0
        mul = 1
        for v in chunk:
            val += int(v + 1) * mul
            mul *= 3
        packed_bytes.append(val & 0xFF)
        if len(chunk) == 5:
            packed_bytes.append((val >> 8) & 0xFF)
        i += 5
    return bytes(packed_bytes)


def unpack_tq1_0(data, count):
    arr = bytearray(data)
    out = np.zeros(count, dtype=np.int8)
    src_idx = 0
    dst_idx = 0
    while dst_idx < count:
        if src_idx + 1 < len(arr):
            val = arr[src_idx] | (arr[src_idx + 1] << 8)
            src_idx += 2
        elif src_idx < len(arr):
            val = arr[src_idx]
            src_idx += 1
        else:
            val = 0
        chunk_size = min(5, count - dst_idx)
        for j in range(chunk_size):
            rem = val % 3
            val //= 3
            out[dst_idx + j] = np.int8(rem - 1)
        dst_idx += chunk_size
    return out
