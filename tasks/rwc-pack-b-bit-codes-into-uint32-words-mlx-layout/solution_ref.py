import numpy as np


def pack_4bit_codes(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint8)
    out = np.zeros(codes.size // 8, dtype=np.uint32)
    for i in range(out.size):
        base = i * 8
        word = np.uint32(0)
        for j in range(8):
            word |= np.uint32(int(codes[base + j]) & 0xF) << np.uint32(4 * j)
        out[i] = word
    return out


def unpack_4bit_codes(words: np.ndarray) -> np.ndarray:
    words = np.asarray(words, dtype=np.uint32)
    out = np.zeros(words.size * 8, dtype=np.uint8)
    for i, word in enumerate(words):
        value = int(word)
        for j in range(8):
            out[i * 8 + j] = np.uint8((value >> (4 * j)) & 0xF)
    return out
