import ref
import numpy as np


def check(workdir):
    out = {"bytes_matched": 0.0}
    try:
        from q4k.quant import round_trip_q4_k, quantize_q4_k_superblock, dequantize_q4_k_superblock
    except Exception as e:
        out["_note"] = f"import error: {e}"
        return out
    w = ref.generate_superblock()
    try:
        ref_b, ref_dec = ref.round_trip_q4_k(w)
        got_b, got_dec = round_trip_q4_k(w)
        if got_b == ref_b:
            out["bytes_matched"] = 1.0
        else:
            out["_note"] = f"serialized bytes differ: len got {len(got_b)}, ref {len(ref_b)}"
    except Exception as e:
        out["_note"] = f"execution error: {e}"
    return out
