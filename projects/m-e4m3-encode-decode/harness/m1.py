import numpy as np
import ref


def check(workdir):
    from fp8.e4m3 import decode_e4m3, encode_e4m3

    out = {"encode_matched": 0.0, "decode_matched": 0.0}

    all_bytes = np.arange(256, dtype=np.uint8)
    ref_decoded = ref.decode_e4m3(all_bytes)
    try:
        got_decoded = decode_e4m3(all_bytes)
        if np.allclose(got_decoded, ref_decoded, equal_nan=True):
            out["decode_matched"] = 1.0
        else:
            out["_note"] = "decode_e4m3 output mismatched reference LUT"
    except Exception as e:
        out["_note"] = f"decode_e4m3 raised exception: {e}"
        return out

    test_inputs = [
        np.array([-500.0, -448.0, -1.0, 0.0, 0.001, 1.0, 448.0, 500.0], dtype=np.float32),
        np.array([np.nan, 0.15625, 128.0, -280.0], dtype=np.float32),
    ]

    enc_ok = True
    for x in test_inputs:
        ref_enc = ref.encode_e4m3(x)
        got_enc = encode_e4m3(x)
        if not np.array_equal(ref_enc, got_enc):
            enc_ok = False
            out["_note"] = f"encode_e4m3 mismatch: got {got_enc}, expected {ref_enc}"
            break

    if enc_ok:
        out["encode_matched"] = 1.0

    return out
