import ref
from calib.entropy import compute_entropy_scale
from reference.calib.entropy import compute_entropy_scale as ref_entropy


def check(workdir):
    out = {"entropy_matched": 0.0}
    ok = 0
    for t in ref.TENSORS:
        want = ref_entropy(t)
        got = compute_entropy_scale(t)
        if abs(got - want) < 1e-3:
            ok += 1
    out["entropy_matched"] = float(ok)
    return out
