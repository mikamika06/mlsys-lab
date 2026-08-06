import torch
import ref


def check(workdir):
    from distill.analysis import softmax_entropy_curve

    torch.manual_seed(42)
    logits = torch.randn(4, 16)
    temperatures = [0.5, 1.0, 2.0, 5.0, 10.0]

    want = ref.ref_softmax_entropy_curve(logits, temperatures)
    try:
        got = softmax_entropy_curve(logits, temperatures)
    except Exception as e:
        return {"entropy_matched": 0.0, "_note": f"raised {type(e).__name__}: {e}"}

    if not isinstance(got, (list, tuple)) or len(got) != len(temperatures):
        return {"entropy_matched": 0.0, "_note": "return type must be a list matching temperature length"}

    matched = 0
    for w, g in zip(want, got):
        if abs(w - float(g)) < 1e-4:
            matched += 1

    return {"entropy_matched": float(matched)}
