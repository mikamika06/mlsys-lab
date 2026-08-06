import ref
import numpy as np


def check(workdir):
    from famask.generator import generate_causal_mask

    out = {"masks_matched": 0.0}
    ok = 0
    for i, (sq, sk, alignment) in enumerate(ref.TEST_CASES):
        want = ref.ref_generate_causal_mask(sq, sk, alignment)
        try:
            got = generate_causal_mask(sq, sk, alignment=alignment)
            got_arr = np.array(got)
            if got_arr.shape == want.shape and np.array_equal(got_arr, want):
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"test case {i} (sq={sq}, sk={sk}, alg={alignment}) mismatch"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"test case {i} error: {type(e).__name__}: {str(e)[:100]}"
    out["masks_matched"] = float(ok)
    return out
