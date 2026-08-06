import torch
import ref


def check(workdir):
    from flowcheck.cond import conditional_branch_fn

    x = torch.tensor([1.0, -1.0])
    res = ref.check_cond_fn(conditional_branch_fn, x)
    exported = res["exported"]
    out = {"exported_cleanly": 1.0 if exported else 0.0}
    if not exported:
        out["_note"] = "Function failed torch.export or did not produce valid outputs"
    return out
