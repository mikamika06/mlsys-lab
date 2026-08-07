import ref
import numpy as np


def check(workdir):
    from flashwrap.recover import recover_denominator

    cases = ref.generate_test_cases()
    matched = 0
    for lse, want in cases:
        try:
            got = recover_denominator(lse)
            if np.allclose(got, want, atol=1e-5, rtol=1e-5):
                matched += 1
        except Exception:
            pass
    return {"denominator_matched": float(matched)}
