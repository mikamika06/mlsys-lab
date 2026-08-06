import numpy as np


def check(workdir):
    from quant.blocks import decode_q4_block, encode_q4_block

    out = {"decoding_matched": 0.0}
    try:
        original = np.array([1, 15, 8, 4, 10, 2], dtype=np.uint8)
        encoded = encode_q4_block(original, "Q4_0")
        decoded = decode_q4_block(encoded, "Q4_0")
        if np.array_equal(original, decoded):
            out["decoding_matched"] = 1.0
        else:
            out["_note"] = f"decoded {decoded} does not match original {original}"
    except Exception as e:
        out["_note"] = f"error in decode/encode: {type(e).__name__}: {e}"
    return out
