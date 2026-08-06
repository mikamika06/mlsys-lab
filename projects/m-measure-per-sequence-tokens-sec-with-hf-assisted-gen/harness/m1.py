import ref
import sys
import os

def check(workdir):
    sys.path.insert(0, workdir)
    import specbatch.measure as measure

    out = {"matched": 0.0}
    ok = 0
    for trace, bs in ref.TRACES:
        want = ref.measure_tokens_per_sec(trace, bs)
        try:
            got = measure.measure_tokens_per_sec(trace, bs)
        except NotImplementedError:
            continue

        if abs(want - got) < 1e-5:
            ok += 1
        else:
            if "_note" not in out:
                out["_note"] = f"trace {trace} bs {bs}: got {got}, want {want}"

    out["matched"] = float(ok)
    return out
