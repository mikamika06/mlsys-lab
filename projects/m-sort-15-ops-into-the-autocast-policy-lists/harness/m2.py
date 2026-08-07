import ref
import torch


def check(workdir):
    from ampcheck.custom_op import SafeCustomOp

    out = {"custom_op_correct": 0.0}
    try:
        x = torch.randn(4, 4, dtype=torch.float32, requires_grad=True)
        w = torch.randn(4, 4, dtype=torch.float32, requires_grad=True)

        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            out_custom = SafeCustomOp.apply(x, w)
            loss = out_custom.sum()
            loss.backward()

        if out_custom.dtype == torch.bfloat16 and x.grad is not None:
            out["custom_op_correct"] = 1.0
        else:
            out["_note"] = f"dtype {out_custom.dtype}, grad {x.grad is not None}"
    except Exception as e:
        out["_note"] = f"error testing custom op: {str(e)[:120]}"
    return out
