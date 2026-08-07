import ref
import torch


def check(workdir):
    from optbits.train import train_short_loop

    torch.manual_seed(123)
    model = ref.Model()
    x = torch.randn(16, 32)
    y = torch.randn(16, 1)

    try:
        init_l, final_l = train_short_loop(model, x, y, steps=5)
    except Exception as e:
        return {
            "converged": 0.0,
            "_note": f"train_short_loop failed: {type(e).__name__}: {str(e)[:100]}",
        }

    out = {"converged": 0.0}
    if final_l < init_l:
        out["converged"] = 1.0
    else:
        out["_note"] = (
            f"loss did not decrease: initial={init_l}, final={final_l}"
        )
    return out
