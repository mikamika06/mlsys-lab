import ref
import numpy as np

def check(workdir):
    from fp8codec.codec import encode_e4m3, decode_e4m3
    tensors = ref.get_test_tensors()
    ok = 0
    for i, x in enumerate(tensors):
        scale = 0.5
        want_enc = ref.encode_e4m3(x, scale)
        want_dec = ref.decode_e4m3(want_enc, scale)
        try:
            got_enc = encode_e4m3(x, scale)
            got_dec = decode_e4m3(got_enc, scale)
            if np.allclose(got_enc, want_enc, atol=1e-3) and np.allclose(got_dec, want_dec, atol=1e-3):
                ok += 1
        except Exception:
            pass
    matched = 1.0 if ok == len(tensors) else 0.0
    return {"encode_decode_matched": matched}
