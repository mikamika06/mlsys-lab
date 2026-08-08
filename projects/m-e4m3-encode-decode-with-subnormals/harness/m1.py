import ref
import numpy as np


def check(workdir):
    from quantlib.codec import encode_e4m3, decode_e4m3
    data = ref.generate_test_data()
    want_enc = ref.encode_e4m3(data)
    got_enc = encode_e4m3(data)
    want_dec = ref.decode_e4m3(want_enc)
    got_dec = decode_e4m3(got_enc)
    match = float(np.mean(want_dec == got_dec))
    out = {"byte_exact_fraction": match}
    if match < 1.0:
        out["_note"] = f"decode mismatch: match fraction {match}"
    return out
