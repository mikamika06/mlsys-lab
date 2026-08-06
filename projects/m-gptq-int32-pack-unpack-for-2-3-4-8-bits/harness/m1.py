import numpy as np
import ref

def check(workdir):
    from quantpack import pack_weights, unpack_weights
    match_count = 0
    total = len(ref.TEST_CASES)
    for weights, bits in ref.TEST_CASES:
        try:
            packed = pack_weights(weights, bits)
            unpacked = unpack_weights(packed, bits, len(weights))
            if np.array_equal(weights, unpacked):
                match_count += 1
        except Exception:
            pass
    fraction = float(match_count) / float(total)
    out = {"byte_exact_fraction": fraction}
    if fraction < 1.0:
        out["_note"] = f"pack/unpack matched {match_count}/{total}"
    return out
