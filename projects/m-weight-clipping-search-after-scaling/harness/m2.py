import numpy as np
import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from awq_clip.quant import search_clipping

    out = {"argmin_index_match": 0.0, "opt_max_match": 0.0}

    try:
        want_idx, want_max = ref.search_clipping_ref(ref.W_TEST, ref.N_BITS, ref.GROUP_SIZE, ref.N_GRID)
        got_idx, got_max = search_clipping(ref.W_TEST, ref.N_BITS, ref.GROUP_SIZE, ref.N_GRID)

        if np.array_equal(got_idx, want_idx):
            out["argmin_index_match"] = 1.0
        else:
            out["_note"] = f"Indices mismatch. Got {got_idx[:5]}, want {want_idx[:5]}"

        if np.allclose(got_max, want_max, atol=1e-5):
            out["opt_max_match"] = 1.0

    except Exception as e:
        out["_note"] = f"Failed to execute search_clipping: {e}"

    return out
