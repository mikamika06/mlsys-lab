import numpy as np
import ref


def check(workdir):
    out = {"scale_correct": 0.0}
    try:
        from fp4quant.scale import quantize_e8m0
    except Exception as e:
        out["_note"] = f"Failed to import quantize_e8m0: {e}"
        return out

    np.random.seed(456)
    boundary_scales = np.array([2**0.5, 2**1.5, 2**2.5, 1.0, 2.0, 4.0, 0.125, 0.35], dtype=np.float32)
    rand_scales = np.random.uniform(0.01, 100.0, size=(50,)).astype(np.float32)
    test_scales = np.concatenate([boundary_scales, rand_scales])

    want = ref.reference_quantize_e8m0(test_scales)
    try:
        got = quantize_e8m0(test_scales)
    except Exception as e:
        out["_note"] = f"quantize_e8m0 raised exception: {e}"
        return out

    if not isinstance(got, np.ndarray) or got.shape != want.shape or not np.array_equal(got, want):
        out["_note"] = f"E8M0 quantization mismatch."
        return out

    out["scale_correct"] = 1.0
    return out
