import numpy as np
import ref


def check(workdir):
    from imatrix.derive import compute_required_tokens
    cases = ref.generate_test_cases()
    ok = 1
    for c in cases:
        want = int(np.ceil(c["variance"] / (c["target_se"] ** 2)))
        try:
            got = compute_required_tokens(c["variance"], c["target_se"])
            if got != want:
                ok = 0
                break
        except Exception:
            ok = 0
            break
    return {"tokens_derived_match": float(ok)}
