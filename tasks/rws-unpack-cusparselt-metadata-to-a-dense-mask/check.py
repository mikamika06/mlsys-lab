import numpy as np

def _pack_mask(mask: np.ndarray) -> np.ndarray:
    flat = mask.ravel()
    n = len(flat)
    pad_len = (-n) % 4
    if pad_len:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=bool)])
    bytes_arr = []
    for i in range(0, len(flat), 4):
        v0, v1, v2, v3 = flat[i:i+4].astype(np.uint8)
        byte = int(v0 | (v1 << 2) | (v2 << 4) | (v3 << 6))
        bytes_arr.append(byte)
    return np.array(bytes_arr, dtype=np.uint8)

def grade(sol, fx) -> dict:
    ok = 1.0
    rng = np.random.default_rng(42)
    shapes = [(5,7), (10,10), (3,12)]
    for shape in shapes:
        mask = rng.integers(2, size=shape, dtype=bool)
        metadata = _pack_mask(mask)
        try:
            out = sol.unpack_cusparselt_metadata(metadata, shape)
        except Exception:
            return {"exact_match": 0.0}
        if not np.array_equal(out, mask):
            return {"exact_match": 0.0}
    return {"exact_match": ok}
