import torch

def verify_skipped_step(model, optimizer, data_tensor):
    param_before = [p.clone() for p in model.parameters()]
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    optimizer.zero_grad()
    with torch.amp.autocast("cpu", dtype=torch.float16):
        out = model(data_tensor)
        loss = out.sum()
    scaler.scale(loss).backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.fill_(float("inf"))
    scaler.step(optimizer)
    scaler.update()
    param_after = [p.clone() for p in model.parameters()]
    unchanged = all(torch.equal(b, a) for b, a in zip(param_before, param_after))
    return unchanged
