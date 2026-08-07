import ref
import numpy as np


def check(workdir):
    from palettize.decode import decode_weight
    from palettize.scale import per_channel_scale
    out = {"decode_match": 0.0}
    try:
        q, s, z = per_channel_scale(ref.TEST_WEIGHT, 4)
        dec = decode_weight(q, s, z)
        ref_dec = (q.astype(np.float32) - z) * s
        if np.allclose(dec, ref_dec, atol=1e-5):
            out["decode_match"] = 1.0
        else:
            out["_note"] = "decoded weight does not match expected formula"
    except Exception as e:
        out["_note"] = f"Error: {e}"
    return out
