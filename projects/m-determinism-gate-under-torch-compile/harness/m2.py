import ref
import torch


def check(workdir):
    from detgate.core import stabilized_gate

    out = {"warmup_matched": 0.0}
    ok = True
    for model, inputs in ref.MODELS:
        compiled = torch.compile(model)
        want = ref.stabilized_gate(compiled, inputs, warmup_runs=1, test_runs=3)
        got = stabilized_gate(compiled, inputs, warmup_runs=1, test_runs=3)
        if bool(want) != bool(got):
            ok = False
            break
    if ok:
        out["warmup_matched"] = 1.0
    return out
