import numpy as np
import ref


def check(workdir):
    from triton_bounds.ops import safe_vector_add

    out = {"correct_outputs": 0.0}
    cases = ref.get_test_cases()
    passed = 0
    for x, y, n in cases:
        try:
            res = safe_vector_add(x, y, n, block_size=64)
            expected = ref.ref_vector_add(x, y, n)
            if np.allclose(res[:n], expected, rtol=1e-5, atol=1e-5):
                passed += 1
            else:
                out["_note"] = f"Mismatch at N={n}"
                break
        except Exception as e:
            out["_note"] = f"Error at N={n}: {type(e).__name__}: {str(e)[:100]}"
            break

    if passed == len(cases):
        out["correct_outputs"] = 1.0
    return out
