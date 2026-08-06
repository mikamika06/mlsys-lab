import ref
import torch


def check(workdir):
    from detgate.core import check_determinism

    out = {"determinism_matched": 0.0}
    ok = True
    for model, inputs in ref.MODELS:
        compiled = torch.compile(model)
        want = ref.check_determinism(compiled, inputs, num_runs=4)
        got = check_determinism(compiled, inputs, num_runs=4)
        if bool(want) != bool(got):
            ok = False
            break
    if ok:
        out["determinism_matched"] = 1.0
    return out
