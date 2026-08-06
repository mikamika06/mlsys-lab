import numpy as np
import ref


def check(workdir):
    from fp8kv.quant import encode_e4m3, decode_e4m3

    out = {"byte_exact_fraction": 0.0}

    all_u8 = ref.generate_e4m3_test_cases()

    import sys
    import os
    sys.path.insert(0, os.path.join(workdir, "reference"))
    import fp8kv.quant as ref_quant

    ref_decoded = ref_quant.decode_e4m3(all_u8)

    try:
        got_decoded = decode_e4m3(all_u8)
    except Exception as e:
        out["_note"] = f"decode_e4m3 raised {type(e).__name__}: {e}"
        return out

    valid_mask = ~np.isnan(ref_decoded)
    match_dec = np.allclose(ref_decoded[valid_mask], got_decoded[valid_mask], rtol=1e-5, atol=1e-5)
    match_nan = np.all(np.isnan(ref_decoded) == np.isnan(got_decoded))

    if not (match_dec and match_nan):
        out["_note"] = "decode_e4m3 output does not match expected values for full uint8 range"
        return out

    test_floats = np.array([
        0.0, -0.0, 0.25, 0.333, 1.0, 1.5, 2.0, 448.0, -448.0, 500.0, -500.0,
        0.001953125, 0.0009765625
    ], dtype=np.float32)

    ref_encoded = ref_quant.encode_e4m3(test_floats)
    try:
        got_encoded = encode_e4m3(test_floats)
    except Exception as e:
        out["_note"] = f"encode_e4m3 raised {type(e).__name__}: {e}"
        return out

    correct_bytes = np.sum(ref_encoded == got_encoded)
    total_bytes = len(test_floats)

    out["byte_exact_fraction"] = float(correct_bytes) / float(total_bytes)
    return out
