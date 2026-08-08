import ref
import numpy as np

def check(workdir):
    from quantpack.packing import pack_weights, unpack_weights
    test_cases = ref.get_test_matrices()
    ok = 0
    total = len(test_cases)
    for mat, bits in test_cases:
        packed_ref = ref.pack_weights(mat, bits)
        unpacked_ref = ref.unpack_weights(packed_ref, bits, mat.shape)
        try:
            packed_got = pack_weights(mat, bits)
            unpacked_got = unpack_weights(packed_got, bits, mat.shape)
            if np.array_equal(unpacked_ref, unpacked_got):
                ok += 1
        except Exception:
            pass
    matched = 1.0 if ok == total else 0.0
    return {"pack_unpack_match": matched}
