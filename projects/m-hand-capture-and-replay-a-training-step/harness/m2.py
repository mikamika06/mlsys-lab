import torch
import ref


def check(workdir):
    out = {"max_abs_err": 1.0}
    if not torch.cuda.is_available():
        out["_note"] = "CUDA not available"
        return out

    torch.manual_seed(42)
    x = torch.randn(4, 8, device="cuda")
    y = torch.randn(4, 8, device="cuda")
    x2 = x + 0.1
    y2 = y + 0.1

    try:
        from capturer.step import CapturedStep
        torch.manual_seed(42)
        model = ref.DummyModel().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        step = CapturedStep(model, optimizer, loss_fn)
        step.capture(x, y)
        got_out, got_loss = step.replay(x2, y2)

        torch.manual_seed(42)
        ref_out, ref_loss = ref.get_reference_step_output(x, y)

        err = torch.max(torch.abs(got_out - ref_out)).item()
        out["max_abs_err"] = float(err)
    except Exception as e:
        out["_note"] = f"Replay failed: {type(e).__name__}: {str(e)[:120]}"

    return out
