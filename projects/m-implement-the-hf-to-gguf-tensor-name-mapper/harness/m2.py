import numpy as np
import ref


def check(workdir):
    from ggufmap.rope import undo_rope_permutation

    out = {"cases_matched": 0.0, "total_cases": float(len(ref.ROPE_TEST_CASES)), "exact_match": 0.0}
    ok = 0
    for i, case in enumerate(ref.ROPE_TEST_CASES):
        tensor = case["tensor"]
        n_heads = case["n_heads"]
        want = ref.undo_rope_permutation(tensor, n_heads)
        try:
            got = undo_rope_permutation(tensor, n_heads)
            if np.array_equal(got, want):
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"case {i}: array shapes got {got.shape}, want {want.shape}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"case {i} raised {type(e).__name__}: {e}"

    out["cases_matched"] = float(ok)
    if ok == len(ref.ROPE_TEST_CASES):
        out["exact_match"] = 1.0
    return out
