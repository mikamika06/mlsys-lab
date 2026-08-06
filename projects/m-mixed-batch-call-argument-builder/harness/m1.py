import ref
import numpy as np

def check(workdir):
    from batch.builder import build_arguments
    out = {"arguments_matched": 0.0}
    ok = 0
    for reqs in ref.REQUESTS_LIST:
        want = ref.build_arguments(reqs)
        got = build_arguments(reqs)
        if got is not None and "max_seqlen" in got and got["max_seqlen"] == want["max_seqlen"]:
            if np.array_equal(got["cu_seqlens"], want["cu_seqlens"]):
                ok += 1
    out["arguments_matched"] = float(ok)
    return out
