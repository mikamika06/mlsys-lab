import torch
import ref


def check(workdir):
    out = {"steps_matched": 0.0}
    if not torch.cuda.is_available():
        out["_note"] = "CUDA not available"
        return out

    torch.manual_seed(42)
    x = torch.randn(4, 8, device="cuda")
    y = torch.randn(4, 8, device="cuda")

    try:
        from capturer.step import CapturedStep
        model = ref.DummyModel().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        step = CapturedStep(model, optimizer, loss_fn)
        step.capture(x, y)
        out["steps_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"Capture failed: {type(e).__name__}: {str(e)[:120]}"

    return out
