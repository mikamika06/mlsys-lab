import numpy as np
import ref
from fp8util.quant import encode_e4m3, decode_e4m3


def check(workdir):
    out = {"encode_decode_exact": 0.0}
    data = ref.generate_test_data(seed=123)

    try:
        enc = encode_e4m3(data)
        dec = decode_e4m3(enc)

        ref_enc = ref.encode_e4m3 if hasattr(ref, 'encode_e4m3') else None
        from reference.fp8util.quant import encode_e4m3 as ref_encode, decode_e4m3 as ref_decode

        r_enc = ref_encode(data)
        r_dec = ref_decode(r_enc)

        if np.array_equal(enc, r_enc) and np.allclose(dec, r_dec, atol=1e-3):
            out["encode_decode_exact"] = 1.0
        else:
            out["_note"] = "Encoded bytes or decoded values do not match reference implementation."
    except Exception as e:
        out["_note"] = f"Error during execution: {type(e).__name__}: {str(e)}"

    return out
