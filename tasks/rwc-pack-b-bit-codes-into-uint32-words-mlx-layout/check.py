import numpy as np


def _oracle_pack(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    out = np.zeros(len(codes) // 8, dtype=np.uint32)
    for i in range(len(out)):
        word = np.uint32(0)
        base = i * 8
        for j in range(8):
            word |= np.uint32(int(codes[base + j]) & 0xF) << np.uint32(4 * j)
        out[i] = word
    return out


def _oracle_unpack(words):
    words = np.asarray(words, dtype=np.uint32)
    out = np.zeros(len(words) * 8, dtype=np.uint8)
    for i, word in enumerate(words):
        for j in range(8):
            out[i * 8 + j] = np.uint8((int(word) >> (4 * j)) & 0xF)
    return out


def grade(sol, fx) -> dict:
    cases = [
        np.arange(64, dtype=np.uint8) % 16,
        np.array([(i * 7 + 3) % 16 for i in range(128)], dtype=np.uint8),
        np.array([15 - (i % 16) for i in range(192)], dtype=np.uint8),
    ]

    ok = 1.0
    for codes in cases:
        try:
            packed = sol.pack_4bit_codes(codes)
            restored = sol.unpack_4bit_codes(packed)
            packed = np.asarray(packed)
            restored = np.asarray(restored)
        except Exception:
            ok = 0.0
            break

        expected = _oracle_pack(codes)
        if packed.dtype != np.uint32 or not np.array_equal(packed, expected):
            ok = 0.0
            break

        expected_codes = _oracle_unpack(expected)
        if restored.dtype != np.uint8 or not np.array_equal(restored, expected_codes):
            ok = 0.0
            break

    return {"exact_match": ok}
