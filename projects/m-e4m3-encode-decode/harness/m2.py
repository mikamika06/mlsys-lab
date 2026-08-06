import ref
import numpy as np

def check(workdir):
    from fp8codec.scale import descale_tensor
    tensors = ref.get_test_tensors()
    ok = 0
    for i, x in enumerate(tensors):
        scale = 2.0
        q = ref.encode_e4m3(x, scale)
        want = ref.descale_tensor(q, scale)
        try:
            got = descale_tensor(q, scale)
            if np.allclose(got, want, atol=1e-3):
                ok += 1
        except Exception:
            pass
    matched = 1.0 if ok == len(tensors) else 0.0
    return {"descale_matched": matched}
