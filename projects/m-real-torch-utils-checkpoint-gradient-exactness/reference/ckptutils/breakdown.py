import torch
from torch.utils.checkpoint import checkpoint


def analyze_op_breakdown(model, inputs, strategy):
    ops_logged = []
    x = inputs.clone().detach().requires_grad_(True)
    out = x
    for i, layer in enumerate(model):
        chk = strategy[i] if i < len(strategy) else False
        ops_logged.append({"layer": i, "checkpointed": chk, "type": type(layer).__name__})
        if chk:
            out = checkpoint(layer, out, use_reentrant=False)
        else:
            out = layer(out)
    loss = out.sum()
    loss.backward()

    total_layers = len(model)
    ckpt_count = sum(1 for s in strategy if s)
    return {
        "ops": ops_logged,
        "total_layers": total_layers,
        "checkpointed_count": ckpt_count,
        "ratio": float(ckpt_count) / float(max(1, total_layers))
    }
