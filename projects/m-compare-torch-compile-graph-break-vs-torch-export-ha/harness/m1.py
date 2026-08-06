import torch
import ref


def check(workdir):
    from flowcheck.analysis import analyze_behavior

    x = torch.tensor([1.0, 2.0])
    got = analyze_behavior(ref.sample_branch_fn, x)
    want = ref.check_analysis(ref.sample_branch_fn, x)

    matched = 1.0 if got == want else 0.0
    out = {"behavior_matched": matched}
    if matched == 0.0:
        out["_note"] = f"got {got}, want {want}"
    return out
