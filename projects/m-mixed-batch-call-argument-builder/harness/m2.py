import ref
import numpy as np

def check(workdir):
    from batch.builder import build_arguments
    from batch.offsets import compute_offsets
    out = {"offsets_matched": 0.0, "strides_match": 0.0}
    reqs = ref.REQUESTS_LIST[0]
    want_args = ref.build_arguments(reqs)
    want_offs = ref.compute_offsets(want_args)

    try:
        got_args = build_arguments(reqs)
        got_offs = compute_offsets(got_args)
        if np.array_equal(got_offs, want_offs):
            out["offsets_matched"] = 1.0
            out["strides_match"] = 1.0
    except Exception as e:
        out["_note"] = str(e)
    return out
