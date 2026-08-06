import ref
import torch


def check(workdir):
    from softmaxln.softmax import fused_softmax

    out = {"rel_err": 0.0}
    torch.manual_seed(1337)
    x = torch.randn(32, 256) * 15.0
    try:
        got = fused_softmax(x)
        want = ref.ref_softmax(x)
        err = torch.max(torch.abs(got - want)).item()
        out["rel_err"] = float(err)
    except Exception as e:
        out["_note"] = str(e)
        out["rel_err"] = 1.0
    return out
