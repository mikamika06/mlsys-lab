import torch


def check(workdir):
    from softmaxln.overflow import naive_softmax_overflow

    out = {"overflow_detected": 0.0}
    x = torch.tensor([[1000.0, 2000.0, 3000.0]])
    try:
        got = naive_softmax_overflow(x)
        if torch.isnan(got).any() or torch.isinf(got).any():
            out["overflow_detected"] = 1.0
        else:
            out["_note"] = "did not overflow"
    except Exception as e:
        out["overflow_detected"] = 1.0
    return out
