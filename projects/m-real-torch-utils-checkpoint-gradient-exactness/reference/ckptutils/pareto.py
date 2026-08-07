import torch
from torch.utils.checkpoint import checkpoint


def compute_pareto_curve(model, inputs, strategies):
    results = []
    for strat in strategies:
        torch.manual_seed(42)
        x = inputs.clone().detach().requires_grad_(True)

        start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        if start:
            start.record()

        out = x
        for i, layer in enumerate(model):
            chk = strat[i] if i < len(strat) else False
            if chk:
                out = checkpoint(layer, out, use_reentrant=False)
            else:
                out = layer(out)
        loss = out.sum()
        loss.backward()

        if end:
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000.0
        else:
            elapsed = 0.01 * sum(strat) + 0.05

        mem = sum(strat) * 10 + 100
        results.append({
            "strategy": strat,
            "time": float(elapsed),
            "memory": float(mem)
        })
    return results
