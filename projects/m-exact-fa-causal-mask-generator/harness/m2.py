import ref
import numpy as np


def check(workdir):
    from famask.disagreement import disagreement_map

    out = {"disagreement_match": 0.0}
    shapes = [(16, 16), (8, 32), (32, 8), (12, 24)]
    ok = 0
    for sq, sk in shapes:
        want = ref.ref_disagreement_map(sq, sk)
        try:
            got = disagreement_map(sq, sk)
            got_arr = np.array(got)
            if got_arr.shape == want.shape and np.array_equal(got_arr, want):
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"disagreement map mismatch for sq={sq}, sk={sk}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"error in disagreement_map: {type(e).__name__}: {str(e)[:100]}"
    if ok == len(shapes):
        out["disagreement_match"] = 1.0
    return out
