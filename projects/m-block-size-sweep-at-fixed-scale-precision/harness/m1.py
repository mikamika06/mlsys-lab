import numpy as np
import ref


def check(workdir):
    out = {"unpack_correct": 0.0}
    try:
        from fp4quant.unpack import unpack_mxfp4
    except Exception as e:
        out["_note"] = f"Failed to import unpack_mxfp4: {e}"
        return out

    np.random.seed(123)
    test_cases = [
        np.random.randint(0, 256, size=(100,), dtype=np.uint8),
        np.random.randint(0, 256, size=(10, 16), dtype=np.uint8),
    ]

    for packed in test_cases:
        want = ref.reference_unpack(packed)
        try:
            got = unpack_mxfp4(packed)
        except Exception as e:
            out["_note"] = f"unpack_mxfp4 raised exception: {e}"
            return out
        if not isinstance(got, np.ndarray) or got.shape != want.shape or not np.array_equal(got, want):
            out["_note"] = f"Unpack mismatch. Got shape {getattr(got, 'shape', None)}, expected {want.shape}"
            return out

    out["unpack_correct"] = 1.0
    return out
