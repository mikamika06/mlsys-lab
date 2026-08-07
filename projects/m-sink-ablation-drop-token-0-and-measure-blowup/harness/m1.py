import ref

def check(workdir):
    import numpy as np

    out = {"mask_match": 0.0}
    try:
        from sink_ablate.mask import reconstruct_mask
        want = ref.reconstruct_mask(ref.NL, ref.NH, ref.SL, ref.DUMP)
        got = reconstruct_mask(ref.NL, ref.NH, ref.SL, ref.DUMP)
        if np.array_equal(want, got):
            out["mask_match"] = 1.0
    except Exception as e:
        out["_note"] = f"{type(e).__name__}: {str(e)}"
    return out
